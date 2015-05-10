require 'nn'
require 'nngraph'
require 'autobw'
require 'math'
require 'optim'

local n_input = 1
local n_output = 1
local n_hidden = 25
local batch_size = 15
local seq_length = 25

local function make_rnn(n_input, n_hidden, n_output)
	local input = nn.Identity()()
	local prev_state = nn.Identity()()

	local next_state = nn.Sigmoid()(nn.CAddTable()({
		nn.Linear(n_input, n_hidden)(input),
		nn.Linear(n_hidden, n_hidden)(prev_state)
	}))

	local output = nn.Linear(n_hidden, n_output)(next_state)

	return nn.gModule({input, prev_state}, {output, next_state})
end

local model = {
	rnn = make_rnn(n_input, n_hidden, n_output),
	criterion = nn.MSECriterion(),

	start_state = torch.zeros(batch_size, n_hidden),

	tape = autobw.Tape(),

	forward = function(self, inputs, targets)
		local loss = 0
		local next_state = self.start_state

		self.tape:start()

		for i = 1, inputs:size(1) do
			output, next_state = unpack(self.rnn:forward({inputs[i], next_state}))
			loss = loss + self.criterion:forward(output, targets[i])
			self.tape:step()
		end

		self.tape:stop()

		return loss
	end,

	backward = function(self)
		self.tape:backward()
	end,
}

local data = torch.linspace(0, 20*math.pi, 1000):sin():view(-1, 1)

local function next_batch()
	local batch = torch.zeros(seq_length, batch_size, 1)
	local start_idx = torch.Tensor(batch_size):uniform():mul(data:size(1) - seq_length):ceil():long()
	for i = 1, batch_size do
		batch:select(2, i):copy(data:sub(start_idx[i], start_idx[i]+seq_length-1))
	end
	return batch
end

local params, grads = model.rnn:getParameters()
params:uniform(-0.1, 0.1)

local function fopt(x)
	if params ~= x then
		params:copy(x)
	end
	grads:zero()

	local batch = next_batch()
	local inputs = batch:sub(1, batch:size(1)-1)
	local targets = batch:sub(2, batch:size(1))

	local loss = model:forward(inputs, targets)
	model:backward()

	return loss, grads
end

for i = 1, 10000 do
	local _, fx = optim.sgd(fopt, params, {})
	print(fx[1])
end

