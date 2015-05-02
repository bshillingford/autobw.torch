require 'torch'
require 'nn'
require 'autobw'

mytest = {}
tester = torch.Tester()

function mytest.TestErrorWhenMultipleForwards()
    torch.manualSeed(1)
    local tape = autobw.Tape()
    local sigm = nn.Sigmoid()
    local input = torch.randn(10)

    tester:assertError(function()
        tape:begin()
        output = sigm:forward(input)
        output = sigm:forward(output)
        tape:stop()

        tape:backward()
    end, 'multiple forward defaults to error')
end

tester:add(mytest)
tester:run()
