# Bureau

This is yet another multi-agent orchestration framework using
anthropic's agent SDK. Hopefully one that's not too weird. This one is
designed for driving a very classical (some might say anachronistic)
top-down heavily planned green-field software development process.

It's for building a thing from scratch, when the thing is somewhat too
big to just "one shot" it.

So: it runs a bunch of agents, using whatever your API or subscription
is. It only works on anthropic/claude currently. Pull requests
welcome.

You give it a config dir with a file (problem.md) and a bunch of
config files, and a work dir to work in, and it'll make a git repo
there and spawn a bunch of agents and get to work solving problem.md.

It defaults to yolo mode. Run it in a sandbox.

The structure of the agents -- when they run and what they run -- is
what matters. We're trying to accomplish a few things simultaneously
here:

  1. Keep agents focused. Each agent has a thing it's supposed to do
     and really only one thing. It starts with a /clear context and
     ends quickly. It might be writing spec/code/tests but it might
     also be planning, critiquing, revising, or judging other agents'
     work. In any case it does just one thing per agent.

  2. Break down large projects into phases where each phase supports
     the next in refining the work towards a finished product. Or put
     another way: to hopefully give each phase's agents enough
     structure that they don't produce incoherent garbage. The phases
     are intended to map more-or-less to waterfall-style (spec,
     interfaces, impl) or TDD-style (test, impl) sequences. This is
     configurable.
     
  3. Within each phase, break down large tasks using (dynamic,
     agent-driven) hierarchical decomposition so that the agents only
     ever work on a small enough task (planning, writing, critiquing,
     revising) that it fits well in context and they don't lose focus
     or drop stuff. The decomposition should map more-or-less to
     "software abstraction units" like modules and classes and
     functions. This is mildly configurable (currently a depth limit
     and some fixed unit-size hints) but I might change it.

  4. Enable parallelism by dynamic dependency analysis (each task says
     which files it reads and writes) and git worktrees. This will
     chew through credits quickly if you let it, but it goes faster.
     This is configurable.

  5. Keep going until the work is actually done. The control
     constructs are all counters and queues in deterministic
     software. It will not stop early when it gets tired.

  6. Give you some tools to do semi-real software engineering, or at
     least allow you to exercise a degree of caution, to measure twice
     before cutting once. You can add critics and phases, and set the
     decomposition and revision cycle counts high enough, that the
     framework can be extremely thorough in its work. The failure mode
     you are more likely to encounter is "it runs out of credits while
     still discussing exactly what to write". It includes some stuff
     to auto-pause at usage limits and limit its burn rate but that
     is not very well tested. Patches welcome.

  7. Give you some insight into what it's doing. There's a web UI
     that spawns automatically and shows you the task tree and lets
     you look at each agent's inputs and outputs and skip/end them
     early.

Of course, claude wrote this code itself, so any bugs are, well,
whatever. I have no idea how to do attribution or blame in this
era. Somehow this thing came into existence and maybe you'll
like it. Or not.
