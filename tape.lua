local Tape = torch.class("autobw.Tape")

local utils = require'autobw.utils'
local ptr = torch.pointer

-- loops over zip(tbl1, tbl2), applying f to the pairs of *tensors*
-- if tbl1 and tbl2 are tensors, then just apply f
local function zip_foreach(tbl1, tbl2, f)
    if utils.istensor(tbl1) and utils.istensor(tbl2) then
        f(tbl1, tbl2)
    elseif utils.istable(tbl1) and utils.istable(tbl2) then
        assert(#tbl1 == #tbl2)
        for i = 1, #tbl1 do
            if utils.istensor(tbl1[i]) and utils.istensor(tbl2[i]) then
                f(tbl1[i], tbl2[i])
            else
                zip_foreach(tbl1[i], tbl2[i], f)
            end
        end
    else
        error('shouldnt reach here, type mismatch between tbl1 and tbl2?')
    end
end

function Tape:__init()
    self:reset()
end

function Tape:reset()
    -- Clear the internal cache of cloned modules.
    self._clones = {}
    self._next_clone_idx = {}
    collectgarbage()
end

-- given a tensor (e.g. x), returns the adjoint for it
-- given a table of tensors, returns all their adjoints
-- note: shape of return value always the same as its input
-- note2: if any of the output tensors are not in mapping, returns a zero tensor
function Tape:_adjoint(x)
    local mapping = self._x_to_dx

    if utils.istensor(x) then
        return mapping[ptr(x)] or self:_zero(x)
    elseif utils.istable(x) then
        local ret = {}
        for  = 1, #x do
            if utils.istensor(x[i]) then
                ret[i] = mapping[ptr(x[i])] or self:_zero(x[i])
            else
                -- for nested tables of tensors
                ret[i] = self:_adjoint(x[i])
            end
        end
        return ret
    else
        error('shouldnt reach here')
    end
end

-- returns an efficient zero tensor, with same size and type as t
function Tape:_zero(t)
    local tensortype = t:type()
    self._zeros = self._zeros or {}
    self._zeros[tensortype] = self._zeros[tensortype] or t.new(1):zero()

    -- TODO: is it necessary to cache these, or is tensor ctor lightweight enough?
    -- it is definitely the case that memory-wise these are cheap, though
    
    -- note that t:size() returns a new LongStorage each time
    -- t.new(storage, storageOffset, size, stride)
    local storage = self._zeros[tensortype]:storage()
    return t.new(storage, 1, t:size(), t:size():fill(0))
end

function Tape:begin()
    self.tape = {}
    self._next_clone_idx = {}
    self._x_to_dx = {} -- map tensor's ptr to the corresponding dtensor

    self._orig_mod_forward = nn.Module.forward
    nn.Module.forward = function(self_, input)
        local self_ = self:_next_clone(self_)
        local output = self._orig_mod_forward(self_, input)
        self.tape[#self.tape+1] = { module=self_, input=input, output=output }
        return output
    end

    self._orig_crit_forward = nn.Criterion.forward
    nn.Criterion.forward = function(self_, input, target)
        local self_ = self:_next_clone(self_)
        local output = self._orig_crit_forward(self_, input, target)
        self.tape[#self.tape+1] = { criterion=self_, input=input, target=target, output=output }
        return output
    end
end

function Tape:start()
    return self:begin()
end

function Tape:stop()
    nn.Module.forward = self._orig_mod_forward
    nn.Criterion.forward = self._orig_crit_forward
end

function Tape:record(func, ...)
    self:begin()
    local result = {func(...)}
    self:stop()
    return unpack(result)
end

function Tape:backward()
    for i = #self.tape, 1, -1 do
        local o = self.tape[i]
        local dinput
        if o.criterion then
            -- call: gradInput = criterion:backward(input, target)
            dinput = o.criterion:backward(o.input, o.target)
        elseif o.module then
            -- first, prepare "gradOutput" (doutput)
            local doutput = self:_adjoint(o.output)
            -- then call: gradInput = module:backward(input, gradOutput)
            dinput = o.module:backward(o.input, doutput)
        else
            error('internal autobw error: tape contains an non-module/criterion object')
        end

        if dinput then
            zip_foreach(o.input, dinput, function(x, dx)
                assert(self._x_to_dx[ptr(x)] == nil)
                self._x_to_dx[ptr(x)] = dx
            end)
        end
    end
end

function Tape:_next_clone(self_)
    -- If we've already seen this module then we swap it out for a clone that shares parameters.
    -- The user owns the original self_ module, but the tape owns any clones it creates.
    --
    -- Clones are created lazily and re-used in subsequent forward passes that use the same tape.
    local clone_self_ = self_
    local p = ptr(self_)

    if self._clones[p] == nil then
        -- Never seen this module before, start tracking it
        self._next_clone_idx[p] = 2 -- not 1
        self._clones[p] = { self_ }
    else
        -- We've seen this module before, find the next available clone
        local idx = self._next_clone_idx[p] or 1
        clone_self_ = self._clones[p][idx]

        if clone_self_ == nil then
            -- No more clones available, need to make another one
            clone_self_ = utils.shared_clone(self_)
            self._clones[p][idx] = clone_self_
        end

        self._next_clone_idx[p] = idx + 1
    end

    return clone_self_
end
