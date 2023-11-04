# SARSA Learning Algorithm


## AIM

To develop SARSA RL to train an agent in Gym environment for optimal policy learning.

## PROBLEM STATEMENT

### Step 1:
Initialize the Q-table with random values for all state-action pairs.

### Step 2:
Initialize the current state S and choose the initial action A using an epsilon-greedy policy based on the Q-values in the Q-table.

### Step 3:
Repeat until the episode ends and then take action A and observe the next state S' and the reward R.

### Step 4:
Update the Q-value for the current state-action pair (S, A) using the SARSA update rule.

### Step 5:
Update State and Action and repeat the step 3 untill the episodes ends.



## SARSA LEARNING FUNCTION

### NAME :  PRIYADARSHINI R 
### REFERENCE NO : 212220230038 

``` PYTHON3
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # Write your code here
    select_action=lambda state,Q,epsilon:\
    np.argmax(Q[state])\
    if np.random.random() > epsilon\
    else np.random.randint(len(Q[state]))
    alphas = decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,
        n_episodes)
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,
        epsilon_decay_ratio,
        n_episodes
    )
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        next_state,reward,done,_=env.step(action)
        next_action=select_action(next_state,Q,epsilons[e])
        td_target=reward+gamma*Q[next_state][next_action]*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state,action=next_state,next_action
        Q_track[e]=Q
        pi_track.append(np.argmax(Q,axis=1))
      V=np.max(Q,axis=1)
      pi=lambda s: {s:a for s, a in enumerate(np.argmax(Q,axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT:

### Mention the optimal policy, optimal value function , success rate for the optimal policy.

![image](https://github.com/priya672003/sarsa-learning/assets/81132849/f416e8ae-f41b-49bc-86e3-c216827ea325)


### Include plot comparing the state value functions of Monte Carlo method and SARSA learning.

![image](https://github.com/priya672003/sarsa-learning/assets/81132849/16f75875-4b37-4edf-822e-273435dec038)




![image](https://github.com/priya672003/sarsa-learning/assets/81132849/1a50768d-da32-409b-8653-328ca8773dde)

## RESULT:

SARSA learning successfully trained an agent for optimal policy.
