require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'layer.LSTM'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required
  self.input_word_encoding_size = utils.getopt(opt, 'input_word_encoding_size')
  self.input_image_encoding_size = utils.getopt(opt, 'input_image_encoding_size')
  self.word_input_layer = utils.getopt(opt, 'word_input_layer')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.frame_length = utils.getopt(opt, 'frame_length', 10)
  self.caption_size = utils.getopt(opt, 'caption_size', 0)
  local dropout = utils.getopt(opt, 'dropout', 0)
  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length') -- WHAT IS THIS???
  -- create the core lstm network. note +1 for both the START and END tokens
  self.core = LSTM.VC_lstm(self.input_image_encoding_size, self.vocab_size + 1, self.input_word_encoding_size, self.word_input_layer, self.rnn_size, self.num_layers, dropout)
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_word_encoding_size) -- word to embedding???kil
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  self.clones = {self.core}
  self.lookup_tables = {self.lookup_table}
  for t=2,self.seq_length+self.frame_length+1 do
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight')
  end
end

function layer:getModulesList()
  return {self.core, self.lookup_table}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:training() end
  for k,v in pairs(self.lookup_tables) do v:training() end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
end


--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
--[[Rui:
takes a batch of videos and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence (ground truth sentence) length and N is batch (so columns are sequences)
--]]
-- imgs: Tensor(self.frame_length * batch_size * feature_dimension)
function layer:sample(imgs, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  if sample_max == 1 and beam_size > 1 then return self:sample_beam(imgs, opt) end -- indirection for beam search

  local batch_size = imgs:size(2)
  self:_createInitState(batch_size)
  local state = self.init_state

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step
  for t=1,self.seq_length+1+self.frame_length do

    local wt, iwt, imt, sampleLogprobs
    if t <= self.frame_length then
      -- feed in the images
      imt = imgs[t]
      wt = torch.zeros(batch_size, self.input_word_encoding_size):cuda()
    elseif t == self.frame_length + 1 then
      -- feed in the start tokens
      iwt = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      wt = self.lookup_table:forward(iwt)
      imt = torch.zeros(batch_size, self.input_image_encoding_size):cuda()
    else
      imt = torch.zeros(batch_size, self.input_image_encoding_size):cuda()
      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, iwt = torch.max(logprobs, 2)
        iwt = iwt:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        iwt = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, iwt) -- gather the logprobs at sampled positions
        iwt = iwt:view(-1):long() -- and flatten indices for downstream processing
      end
      wt = self.lookup_table:forward(iwt)
    end

    if t >= self.frame_length + 2 then 
      seq[t-self.frame_length-1] = iwt -- record the samples
      seqLogprobs[t-self.frame_length-1] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end

    assert(wt:size(2) == self.input_word_encoding_size, 'word_input:size(2) == self.input_word_encoding_size')
    local inputs = {imt, unpack(state)}
    table.insert(inputs, wt)
    local out = self.core:forward(inputs)
    assert(self.num_state+1 == #out, 'self.num_state+1 == #out')
    logprobs = out[self.num_state+1] -- last element is the output vector
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[
Implements beam search. Really tricky indexing stuff going on inside. 
Not 100% sure it's correct, and hard to fully unit test to satisfaction, but
it seems to work, doesn't crash, gives expected looking outputs, and seems to 
improve performance, so I am declaring this correct.
]]--
function layer:sample_beam(imgs, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 10)
  local batch_size, feat_dim = imgs:size(2), imgs:size(3)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)

  -- lets process every image independently for now, for simplicity
  for k=1,batch_size do
    -- create initial states for all beams
    self:_createInitState(beam_size)
    local state = self.init_state

    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    for t=1,self.seq_length+self.frame_length do

      local wt, iwt, imt, sampleLogprobs
      local new_state
      if t <= self.frame_length then
        -- feed in the images
	imt = imgs[t][k]:expand(beam_size, feat_dim) -- k'th image in the batch
	wt = torch.zeros(beam_size, self.input_word_encoding_size)
        --local imgk = imgs[{ {k,k} }]:expand(beam_size, feat_dim) -- k'th image feature expanded out
        --xt = imgk
      elseif t == self.frame_length+1 then
        -- feed in the start tokens
        iwt = torch.LongTensor(beam_size):fill(self.vocab_size+1)
        wt = self.lookup_table:forward(iwt)
	imt = torch.zeros(beam_size, feat_dim)
      else
        --[[
          perform a beam merge. that is,
          for every previous beam we now many new possibilities to branch out
          we need to resort our beams to maintain the loop invariant of keeping
          the top beam_size most likely sequences.
        ]]--
        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 3 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 3 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-3}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-3}, {} }]:clone()
        end
        for vix=1,beam_size do
          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 3 then
            beam_seq[{ {1,t-3}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-3}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          -- append new end terminal at the end of this beam
          beam_seq[{ t-2, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-2, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

          if v.c == self.vocab_size+1 or t == self.seq_length+2 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end
        
        -- encode as vectors
        iwt = beam_seq[t-2]
        wt = self.lookup_table:forward(iwt)
	imt = torch.zeros(beam_size, feat_dim)
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams

      local inputs = {imt,unpack(state)}
      table.insert(inputs, wt)
      local out = self.core:forward(inputs)
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[Rui:
input is a tuple of:
1. imgs: torch.Tensor of size frame_length x N x K (K is dim of image code; N is the batchSize)
2. seq: torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size and D = opt.seq_length

returns a (D+2)xNx(M+1) Tensor giving (normalized) log probabilities for the 
next token at every iteration of the LSTM (+2 because +1 for first dummy 
img forward, and another +1 because of START/END tokens shift)
--]]
function layer:updateOutput(input)
  local imgs = input[1]
  local seq = input[2]
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

  assert(seq:size(1) == self.seq_length)
  local batch_size = seq:size(2)
  self.output:resize(self.seq_length+1+self.frame_length, batch_size, self.vocab_size+1)
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state}
  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  for t=1,self.seq_length+1+self.frame_length do

    local can_skip = false
    -- local xt
    local wt, iwt, imt
    if t <= self.frame_length then
      -- feed in the images
      imt = imgs[t] -- NxK sized input
      wt = torch.zeros(batch_size, self.input_word_encoding_size):cuda()
    elseif t == self.frame_length + 1 then
      -- feed in the start tokens
      local iwt = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      self.lookup_tables_inputs[t] = iwt
      wt = self.lookup_tables[t]:forward(iwt) -- NxK sized input (token embedding vectors)
      imt = torch.zeros(batch_size, self.input_image_encoding_size):cuda()
    else
      imt = torch.zeros(batch_size, self.input_image_encoding_size):cuda()
      -- feed in the rest of the sequence...
      local iwt = seq[t-self.frame_length-1]:clone()
      if torch.sum(iwt) == 0 then
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true 
      end
      --[[
        seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
        that won't make lookup_table crash with an error.
        token #1 will do, arbitrarily. This will be ignored anyway
        because we will carefully set the loss to zero at these places
        in the criterion, so computation based on this value will be noop for the optimization.
      --]]
      iwt[torch.eq(iwt,0)] = 1

      if not can_skip then
        self.lookup_tables_inputs[t] = iwt
        wt = self.lookup_tables[t]:forward(iwt)
      end
    end

    if not can_skip then
      -- construct the inputs
      self.inputs[t] = {imt,unpack(self.state[t-1])}
      table.insert(self.inputs[t], wt)
      -- forward the network
      local out = self.clones[t]:forward(self.inputs[t])
      -- process the outputs
      self.output[t] = out[self.num_state+1] -- last element is the output vector
      self.state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.state[t], out[i]) end
      self.tmax = t
    end
  end

  return self.output
end


--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
  local dimgs = torch.zeros(self.frame_length, self.caption_size, self.input_image_encoding_size):cuda() -- grad on input images
  --local dwords   -- we don't need the gradient w.r.t. word input

  -- go backwards and lets compute gradients
  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do table.insert(dout, dstate[t][k]) end
    table.insert(dout, gradOutput[t])
    local dinputs = self.clones[t]:backward(self.inputs[t], dout)
    -- split the gradient to xt and to state
    -- dinputs[1] is dimt and dinputs[#dinputs] is dwt(d_word)
    local dimg = dinputs[1] -- first element is the input vector
    --dwords = dinputs[self.num_state+2]
    dstate[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(dstate[t-1], dinputs[k]) end
    
    if t <= self.frame_length then
      dimgs[t] = dimg
    else
      local it = self.lookup_tables_inputs[t]
      self.lookup_tables[t]:backward(it, dimg) -- backprop into lookup table
    end
  end

  -- here only the d w.r.t. imags get propagated back
  self.gradInput = dimgs 
  --{dimgs, torch.Tensor()} we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop

  return self.gradInput
end




-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init(frm_len)
  parent.__init(self)
  self.frame_length = frm_len
end

--[[
input is a Tensor of size (D+2)xNx(M+1)
seq is a LongTensor of size DxN. The way we infer the target in this criterion is as follows:
- at first time step the output is ignored (loss = 0). It's the image tick
- the label sequence "seq" is shifted by one to produce targets
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]

--[[RUI:
input is a Tensor of size (D+1+self.frame_length)xNx(M+1)
seq is a LongTensor of size DxN. The way we infer the target in this criterion is as follows:
- at the first self.frame_length time steps the output is ignored (loss = 0). It's the image sequence ticks
- the label sequence "seq" is shifted by self.frame_length to produce targets
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L, N, Mplus1 = input:size(1), input:size(2), input:size(3)
  local D = seq:size(1)
  assert(D == L-1-self.frame_length, 'input Tensor should be (1+self.frame_length) larger in time')

  local loss = 0
  local n = 0

  for b=1,N do -- iterate over batches
    local first_time = true

    for t=self.frame_length+1,L do -- iterate over sequence time (ignore t=1:self.frame_legnth, dummy forward for the frames)
      -- fetch the index of the next token in the sequence
      local target_index
      if t-self.frame_length > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t-self.frame_length,b}] -- t-1 is correct, since at t=2 START token was fed in and we want to predict first word (and 2-1 = 1).
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mplus1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{t,b,target_index}] -- log(p) since the last layer in LSTM is LogSoftmax
        self.gradInput[{ t,b,target_index }] = -1  
        n = n + 1
      end
    end

  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n)
  return self.output
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
