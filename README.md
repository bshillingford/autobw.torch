# autobw
`autobw` is a simple library for automatically performing **a backwards pass, given only a forwards pass,** in Torch. 
 A major advantage of this is that the neural network's **structure need not be fixed before runtime**. This allows for easy implementation of structures such as recurrent networks. See the example below.


Backpropagation is often described as a method for propagating gradients through a computational graph. One way to implement it for graphs is to explicitly construct a graph given by the user, then evaluate the computational nodes in the order specified in the forward pass, then again but in reverse for the backward pass. 

## Install
```
luarocks install https://raw.githubusercontent.com/bshillingford/autobw.torch/master/autobw-scm-1.rockspec
```

## Details
A method that's closer to how one may reason about a neural network is to explicitly write down a forward pass while recording the statements as they are being executed, then execute the statements' derivative computations (aka adjoint) in reverse. This is equivalent to specifying a computation graph, but more explicit, and allows the user to use **control-flow such as for loops and conditionals**.

This is similar to the approach taken by implementations of reverse-mode automatic differentiation, see e.g. <http://arxiv.org/abs/1502.05767>.

## Examples:
A simple example of computing `linear(x1) + x2 * sigmoid(x3)`, but **randomly** replacing `sigmoid(x3)` with `x3` sometimes:
```lua
lin = nn.Linear()
add = nn.CAddTable()
mul = nn.CMulTable()
sigm = nn.Sigmoid()

tape = autobw.Tape()

-------------- START OF FORWARD PASS --------------
-- records the sequence of operations
tape:begin()
coin_flip = torch.rand()[1]
val1 = lin:forward(x1)

if coin_flip > 0.5 then
  maybe_sigmoid = sigm:forward(x3)
else
  maybe_sigmoid = x3
end

result = add:forward{val1, mul:forward{x2, maybe_sigmoid})
tape:stop()
-------------- END OF FORWARD PASS --------------

-- Play it back in reverse:
tape:backward()

-- Now, the gradients are in the four nn.Module objects as usual.
```

Note: I don't actually use the gradients at all here, and I don't set them to zero first, just to keep the example simple.
See also [our nngraph practical](https://github.com/oxford-cs-ml-2015/practical5/blob/master/practical5.pdf) for the equivalent in `nngraph`.

### LSTM example
The LSTM example <https://github.com/oxford-cs-ml-2015/practical6> can easily be shortened by using this. We delete the backward pass, and simply play it back from the recorded forward pass:
```lua
-- setup autodiff
tape = Tape() -- TODO: local

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    
    ------------------ get minibatch -------------------
    local x, y = loader:next_batch()

    ------------------- forward pass -------------------
    tape:begin() -----------------
    local embeddings = {}            -- input embeddings
    local lstm_c = {[0]=initstate_c} -- internal cell states of LSTM
    local lstm_h = {[0]=initstate_h} -- output values of LSTM
    local predictions = {}           -- softmax outputs
    local loss = 0

    for t=1,opt.seq_length do
        embeddings[t] = clones.embed[t]:forward(x[{{}, t}])

        -- we're feeding the *correct* things in here, alternatively
        -- we could sample from the previous timestep and embed that, but that's
        -- more commonly done for LSTM encoder-decoder models
        lstm_c[t], lstm_h[t] = unpack(clones.lstm[t]:forward{embeddings[t], lstm_c[t-1], lstm_h[t-1]})

        predictions[t] = clones.softmax[t]:forward(lstm_h[t])
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    tape:stop() -----------------

    ------------------ backward pass -------------------
    tape:backward()

    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    initstate_c:copy(lstm_c[#lstm_c])
    initstate_h:copy(lstm_h[#lstm_h])

    -- clip gradient element-wise
    grad_params:clamp(-5, 5)

    return loss, grad_params
end
```
