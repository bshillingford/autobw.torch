local utils = {}

function utils.istensor(x)
    local typename = torch.typename(x)
    return typename and typename:find('Tensor')
end

function utils.istorchclass(x)
    return type(x) == 'table' and torch.typename(x)
end

function utils.istable(x)
    return type(x) == 'table' and not torch.typename(x)
end

function utils.shared_clone(net)
    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)

    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    if net.parameters then
        local cloneParams, cloneGradParams = clone:parameters()
        local cloneParamsNoGrad
        for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
        end
        if paramsNoGrad then
            cloneParamsNoGrad = clone:parametersNoGrad()
            for i =1,#paramsNoGrad do
                cloneParamsNoGrad[i]:set(paramsNoGrad[i])
            end
        end
    end

    mem:close()
    return clone
end



return utils

