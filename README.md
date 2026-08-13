This is my implementation of micrograd which I learned from Andrej Karpathy
# HIGH-LEVEL ARCHITECTURE

```
Value (scalar wrapper)
    ↓
Operations (+, *, **, etc. — each returns a new Value)
    ↓
Computational Graph (implicit, built from Value references)
    ↓
Local Derivative Storage (per-node)
    ↓
Topological Ordering
    ↓
Backpropagation Engine (.backward())
    ↓
Neuron (weighted sum of Values + activation)
    ↓
Layer (collection of Neurons)
    ↓
MLP (stack of Layers)
    ↓
Loss Function (Value-producing comparison)
    ↓
Training Loop (forward → backward → update → zero-grad)
```

## Component responsibilities

**Value** — wraps a scalar and everything needed to later compute a gradient for it. Ask yourself: if `Value` only stored the number itself, would backprop even be possible later? What's the minimum extra state that must ride along with every number the moment it's created?

**Operations** — each operation (add, multiply, etc.) is a *factory* for new `Value` nodes, not a mutation. Ask: why must `a + b` return a brand-new `Value` rather than modifying `a` or `b` in place? What would break if operations mutated their inputs?

**Computational graph** — this isn't a separate class you build — it emerges from the references each `Value` keeps to the values it was made from. Ask: what's the *minimum* set of references a node needs to reconstruct "how did I come to exist" after the fact?

**Local derivative storage** — each node needs to know, locally, how its output changes with respect to each of its immediate inputs. Ask: does a `Value` need to know the derivative of the *entire* graph's output with respect to itself at the moment it's created, or something much smaller and more local?

**Topological ordering** — determines the *sequence* in which backward computation must visit nodes. Ask: if node C depends on both A and B, and you haven't finished accumulating all of C's downstream contributions yet, what could go wrong if you process A's gradient before C's is fully settled?

**Backpropagation engine** — the traversal + accumulation logic that turns local derivatives into full gradients via the chain rule.

**Neuron → Layer → MLP → Loss → Training loop** — these are just increasingly large *compositions* of the Value/operation machinery you already built. If your Value engine is correct, these should require very little new conceptual machinery — mostly organization.

---

# MILESTONES

## Milestone 1 — Core scalar representation

**Goal:** Define what a `Value` is before any math happens.

**Components involved:** `Value` only.

**What it must be capable of doing:** Hold a numeric payload. Be distinguishable from a plain Python float (i.e., something that operations can specifically recognize and act on).

**Important invariants:** A `Value` should always be *inspectable* — you should always be able to ask "what number does this represent right now?"

**Questions to answer yourself:**
- What is the actual data a `Value` needs to store *before* you've done any calculus? (Hint: think about what you'd want printed if you did `print(some_value)`.)
- Should the gradient field exist from the very beginning, or only once backprop is relevant? What would happen if you forgot to initialize it?

**How to test:** Can you create a `Value`, retrieve its number, and print it in a readable form? That's it — nothing calculus-related yet.

---

## Milestone 2 — Forward operations

**Goal:** Make `Value`s combine into new `Value`s via arithmetic.

**Components involved:** `Value`, operator overloading (`__add__`, `__mul__`, etc.).

**What it must be capable of doing:** `a + b` where `a` and `b` are `Value`s should produce a new `Value` whose number is the correct sum.

**Important invariants:** The *inputs* to an operation must remain unmodified after the operation runs.

**Questions to answer yourself:**
- What happens if someone writes `a + 3` where `3` is a plain Python int, not a `Value`? Does your operation need to handle that case, and if so, how do you turn `3` into something your operation can work with?
- Is `a + b` guaranteed to behave the same as `b + a` in your implementation, or could the order matter for reasons beyond the math? (Think about what "reflected" operators like `__radd__` are for.)

**How to test:** Build a handful of expressions using only `+`, `*`, and check the resulting `.data`/number against hand-calculated values. No gradients yet — just verify forward arithmetic is correct.

---

## Milestone 3 — Computational graph (structure)

**Goal:** Make every `Value` aware of *how* it was created.

**Components involved:** `Value`, updated operation logic.

**What it must be capable of doing:** Given any `Value` that resulted from an operation, you should be able to ask "what values were combined to produce you, and by what operation?"

**Important invariants:** A `Value` created directly by the user (a "leaf") should be distinguishable from a `Value` created by an operation.

**Questions to answer yourself:**
- If a `Value` was produced from two other values, what information must the resulting node remember so that you can later trace backward through the graph?
- Should a node remember its parents (the values it came from) or its children (values that later use it)? Which direction does backpropagation actually need to walk?
- What should a leaf node's "parents" look like? Empty? None? Why does that choice matter for how you'll write the traversal later?

**How to test:** Build an expression like `d = a * b + c`. Can you programmatically walk from `d` back to `a`, `b`, and `c` and confirm the graph structure matches what you'd draw on paper?

---

## Milestone 4 — Local derivatives

**Goal:** Attach to each operation the knowledge of how its output changes with respect to its *immediate* inputs only.

**Components involved:** Each operation's implementation.

**What it must be capable of doing:** For `c = a * b`, know that ∂c/∂a = b and ∂c/∂b = a — *without* yet knowing anything about a larger graph these might sit inside.

**Important invariants:** Local derivatives are purely local — they should not reference or assume anything about a global loss or final output.

**Questions to answer yourself:**
- For each operation you plan to support, what is ∂(output)/∂(each input)? Have you derived this by hand before trying to encode it?
- Where should this local-derivative-computing logic live — inside the operation itself, or as a separate function attached to each created node? What are the tradeoffs?
- If an operation has only one input (like a unary function, e.g. `tanh`), does your representation for "local derivatives with respect to inputs" still make sense, or does it need to flex?

**How to test:** Pick a single operation, compute its output for specific numbers, and separately (by hand, on paper) verify the local derivative formula you're about to encode is mathematically correct before you write it.

---

## Milestone 5 — Backpropagation (single path first)

**Goal:** Propagate a gradient from an output back through a *simple, non-branching* chain of operations.

**Components involved:** `Value.grad` field, a backward mechanism per node.

**What it must be capable of doing:** For a graph like `a → b → c` (no branching), starting from `c`'s gradient, correctly compute `a`'s and `b`'s gradients using the chain rule.

**Important invariants:** The gradient of the final output with respect to itself is the multiplicative identity — ask yourself what that value must be, and why starting from anything else would break every downstream calculation.

**Questions to answer yourself:**
- The chain rule says the gradient of a node equals (local derivative) × (gradient of what it feeds into). What piece of information does a node need access to *before* it can compute its own gradient this way?
- If you're processing nodes in the wrong order, what symptom would you observe — a crash, or a silently wrong number?

**How to test:** Build a single-chain expression (no reuse of any intermediate value), manually work out all derivatives on paper, run your backward pass, and compare every single `.grad` value to your hand-worked answer.

---

## Milestone 6 — Topological ordering

**Goal:** Determine the correct node-processing order for graphs that branch — i.e., where a single `Value` is used in more than one downstream computation.

**Components involved:** A graph traversal routine (likely DFS-based) that produces an ordering.

**What it must be capable of doing:** Given a graph, produce a linear ordering of all nodes such that every node appears *after* all the nodes that depend on it — but you need to reason about which direction that ordering should go for backward-mode traversal.

**Important invariants:** Every node must appear exactly once in the final order (watch out for graphs where a node is reachable via multiple paths).

**Questions to answer yourself:**
- If a computation graph contains dependencies like A → C and B → C, in what order must nodes be processed during backpropagation so that C has *already* received all of its incoming gradient contributions before it tries to pass gradient further back to A and B?
- What happens if the same node is visited twice during your traversal, and you don't guard against it — does the ordering become wrong, or does something else break?
- Is this closer to a forward-order or reverse-order topological sort, given that gradients flow from outputs back to inputs?

**How to test:** Build a graph with an intentional diamond shape (some value used twice, like `d = a*a` reusing `a`, or `y = x*x + x`), print your computed ordering, and manually verify no node is processed before something that depends on it should have finished.

---

## Milestone 7 — Full backpropagation (with branching + accumulation)

**Goal:** Correctly backpropagate through graphs where nodes are reused (branching), which requires gradient accumulation.

**Components involved:** `Value.grad`, topological order from Milestone 6, per-node backward step from Milestone 5.

**What it must be capable of doing:** For `y = x*x` (x used twice), correctly get dy/dx = 2x, not x.

**Important invariants:** Every node's gradient, once backward is complete, must reflect the *sum* of its contributions along *every* path from itself to the final output.

**Questions to answer yourself:**
- Why can gradients need to be accumulated rather than simply overwritten? Think concretely about `y = x * x` — how many separate "paths" does x have to y, and what does each path independently contribute?
- If you initialize every node's gradient to zero before starting the backward pass, why does that matter? What would go wrong if you didn't?
- What invariant should hold true across the *entire* graph once backward() has finished running — for every single node, not just the leaves?

**How to test:** Build several graphs with intentional value reuse (`x*x`, `x*x*x`, `x + x`), compute the expected derivative by hand (e.g. d/dx(x²) = 2x), and confirm your engine's `.grad` matches. Then implement or use numerical gradient checking (see Testing section) as a second, independent verification method.

---

## Milestone 8 — Neural network abstractions

**Goal:** Build `Neuron → Layer → MLP` purely by composing your existing `Value` machinery — no new calculus required.

### Neuron
- **Responsibility:** Take a fixed-size input, compute a weighted sum plus bias, apply a nonlinearity.
- **Inputs:** A list of `Value`s (or raw numbers you convert).
- **Outputs:** A single `Value`.
- **Depends on:** Value, operations, an activation function (which you'll need to define as a *new operation* with its own local derivative — think about what that requires from Milestone 4).
- **Test:** Does a manually-constructed neuron with known weights produce the output you'd compute by hand?
- **Question:** Where do the weights themselves live — as plain numbers, or as `Value`s? What breaks if they're not `Value`s?

### Layer
- **Responsibility:** Run several neurons over the same input in parallel.
- **Depends on:** Neuron.
- **Test:** Does a layer of N neurons over an M-dimensional input produce N outputs?

### MLP
- **Responsibility:** Chain multiple layers, feeding one layer's output as the next layer's input.
- **Depends on:** Layer.
- **Test:** Does the output dimensionality match what the layer sizes imply?
- **Question:** How will you gather *all* the weight `Value`s across every neuron in every layer into one flat list — you'll need this for the training loop's update step.

### Loss
- **Responsibility:** Take predictions and targets, produce a single scalar `Value` representing how wrong the network is.
- **Depends on:** Your existing operations (a simple loss can be built entirely from what you already have — think about what operation "difference squared" decomposes into).
- **Question:** Why must the loss be a `Value` and not a plain float?

### Training loop
- **Responsibility:** Repeat: forward pass → compute loss → backward pass → update parameters → reset gradients.
- **Question:** Why must gradients be reset to zero *before* each new backward pass? What would happen across multiple training steps if you forgot this, given what you learned about accumulation in Milestone 7?
- **Question:** What's the actual parameter update rule doing conceptually — why does subtracting (a small step size × gradient) move you toward lower loss rather than higher?

---

# OPERATIONS TO SUPPORT

## Minimum set

**Addition**
- Math: c = a + b
- Inputs: two Values (or Value + scalar)
- Output: one Value
- Derivatives to derive: ∂c/∂a, ∂c/∂b
- Edge cases: adding a `Value` to a raw number — does your code path handle both operand orders?

**Multiplication**
- Math: c = a × b
- Derivatives to derive: ∂c/∂a, ∂c/∂b (think about *why* these involve the other operand)
- Edge cases: multiplying by zero — does the derivative still come out correct?

**Power (exponent)**
- Math: c = a^n (n typically a constant, not a Value)
- Derivatives to derive: ∂c/∂a
- Edge cases: negative or fractional exponents — is your derivative formula still valid? What about a=0 with n<1?

**Negation** (can be derived from multiplication by -1, but ask whether you want it as its own operation)

**Subtraction, division** (ask yourself: can these be *expressed* in terms of operations you already have, rather than implemented from scratch? What does that imply about how much new derivative-derivation work you actually need to do?)

**A nonlinearity — pick one (tanh, ReLU, or sigmoid)**
- Math: whichever you choose
- Derivatives to derive: this is usually the first derivative you'll compute that isn't just "the other operand" — work it out carefully by hand first
- Edge cases: ReLU's derivative at exactly 0 — what will you choose to do there, and can you justify it?

## Optional extensions (after the above are solid)
- `exp`, `log`
- Other activations (sigmoid, leaky ReLU)
- `**` with a `Value` exponent (not just a constant) — think about why this is meaningfully harder than constant-exponent power

---

# BACKPROPAGATION — ARCHITECTURAL DEEP DIVE

**What must be stored in each node?**
Think in terms of three categories: (1) what the node *is* — its numeric value, (2) what the node *came from* — enough to reconstruct the graph, (3) what the node *needs for gradient bookkeeping* — accumulated gradient, and whatever lets it compute its local contribution to its parents once asked.

**What should happen when an operation creates a new node?**
Ask yourself: at creation time, do you already know everything you'll ever need for the backward pass, or is some information only available later (like "what's my gradient")? Which category does each piece of information you identified above fall into?

**Where should the local derivative information live?**
You have two real options: computed on-demand during the backward pass, or precomputed and stored at creation time. Ask: what does each choice cost you, in terms of what data you'd need to keep around from the forward pass?

**How should gradients propagate?**
At each node during the backward walk, you already have that node's own accumulated gradient (how much the final output cares about this node). Ask: given that, and given the local derivative from Milestone 4, how do you compute the contribution to send to *each* parent? What operation combines "how much do I matter" with "how much does my parent matter to me"?

**Why can gradients need to be accumulated?**
Revisit this from Milestone 7 — but now ask it at the architectural level: does *every* node need accumulation logic, or only nodes with more than one downstream consumer? Is it safe to just always accumulate, even for nodes with a single consumer? Why or why not?

**Why is graph traversal order important?**
Ask: what does it mean for a node's gradient to be "finalized"? Can you safely propagate a node's gradient onward to its parents before its own gradient is finalized? What does the topological order from Milestone 6 guarantee about this?

**What happens when a node has multiple downstream paths?**
Each downstream use independently computes a contribution to this node's gradient during the backward walk. Ask: by the time you get around to processing this node's own backward step, have all of its contributions definitely already arrived? What does that depend on?

**What invariant should hold after backpropagation completes?**
Think about this in terms of the mathematical definition of a gradient — for every single node in the graph, what quantity should `.grad` equal, expressed in terms of the final output?

---

# TESTING STRATEGY

1. **Basic arithmetic tests** — build small expressions, check `.data` matches hand-computed forward values. No gradients involved.

2. **Derivative tests (single operation)** — for one isolated operation (e.g. just `c = a * b`), call backward, check `.grad` on `a` and `b` against the hand-derived local formula.

3. **Chain rule tests** — build a short chain (`a → b → c`, no branching), hand-compute the full derivative via the chain rule, compare.

4. **Multi-path graph tests** — build graphs with intentional value reuse (diamonds), hand-derive expected gradients, compare.

5. **Gradient accumulation tests** — specifically construct cases like `y = x + x` and `y = x * x`, where the correct answer is *sensitive* to whether accumulation is implemented correctly (an unaccumulated implementation will give you a subtly wrong, smaller number).

6. **Numerical gradient checking** — the mathematical idea: for any differentiable function f and small ε, the derivative at a point can be approximated by (f(x+ε) − f(x−ε)) / (2ε). Compute this numerical approximation independently of your analytical (chain-rule-based) gradient, for the same inputs, and confirm they agree to within a small tolerance. This is powerful precisely because it doesn't rely on any of the machinery you're trying to verify — it's an independent check. Ask yourself: why is the symmetric difference quotient (using both +ε and −ε) more accurate than a one-sided difference?

7. **Neural network training tests** — construct a tiny, trivially learnable dataset (e.g. a single training example, or a simple linearly separable set), run your training loop for a fixed number of steps, and confirm loss decreases. Ask: if loss doesn't decrease, how will you determine whether the bug is in your autodiff engine, your neuron/layer composition, or your training loop's update step? What's the smallest possible test you could isolate to narrow this down?

---

# SUCCESS CRITERIA CHECKLIST

```
[✅] Scalar value representation works (can create, inspect, print a Value)
[✅] Addition works (forward pass correct)
[✅] Multiplication works (forward pass correct)
[ ] Power operation works (forward pass correct)
[✅ ] Chosen nonlinearity works (forward pass correct)
[✅] Graph structure correctly tracks parent relationships
[✅] Local derivatives correctly computed for each operation (verified by hand)
[✅] Single-path backpropagation produces correct gradients
[✅] Topological ordering correctly handles branching graphs
[ ] Gradient accumulation works for reused values (x*x, x+x cases)
[✅] Full backpropagation matches numerical gradient checking within tolerance
[ ] Neuron produces correct output for known weights
[ ] Layer correctly aggregates multiple neurons
[ ] MLP correctly chains multiple layers
[ ] Loss function produces a differentiable scalar Value
[ ] Parameter collection correctly gathers all weights across the MLP
[ ] Gradients correctly reset to zero between training steps
[ ] Training loop reduces loss over multiple steps on a toy dataset
```
