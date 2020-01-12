# Portfolio-Management-using-Reinforcement-Learning
Portfolio management performed by asset allocation using Reinforcement Learning

### Introduction
 
There is a mix of assets and investment alternatives in the market that needs to be analyzed to make the right decision around the allocation of funds to different assets depending on several objectives that might differ from investor to investor. Though investing in volatile instruments is lucrative in terms of higher expected payoffs, it exposes you to the risk of losing your funds. Investing in Government bonds on the other end is conservative and provides a slower growth to invested funds. Often, the investor is interested in holding emergency funds in the form of money market instruments (cash equivalents). With such varying needs, it’s often wise balancing risk against performance.

Diversification is a risk management strategy that mixes a wide variety of investments within a portfolio. A diversified portfolio contains a mix of distinct asset types and investment vehicles in an attempt at limiting exposure to any single asset or risk. The rationale behind this technique is that a portfolio constructed of different kinds of assets will, on average, yield higher long-term returns and lower the risk of any individual holding or security.
 
While diversifying funds across assets reduce the overall portfolio risk and hedges against market volatility, it limits short term gains. Also, it’s very time consuming to manage allocations across various alternatives present in the market. Besides, the more we diversify more is the transaction fee incurred as it requires rearranging funds depending on market conditions at any particular time.

### Problem Definition

Given the pros and cons of diversifying funds across a portfolio, we seek to understand how the agent acts to allocate income to maximize his/her payoffs within a fixed investment cycle. Our project aims to use Reinforcement Learning to optimize agent’s (investor) decisions to invest funds under several environmental constraints in order to gain maximum cumulative returns over the investment cycle.
We consider four different instruments in our project representative of different investment alternatives with varying risks and payoffs. The same has been listed in increasing order of volatility and expected payoffs here:

Government Bonds: Risk-free asset in our portfolio with an expected payoff of $10
Cash Equivalents: A lower risk instrument (sd: 3.33) with an expected payoff of $23
Real Estate derivatives: A higher risk instrument (sd: 6.7) with an expected payoff of $37
Equities: Highly volatile instrument with a high expected payoff of $50.

The above choices to expected payoffs and risks are not based on any market data and are solely based on simulation. Refer to Dataset generation for additional details on the data simulation results.
 
### Literature Review 

To gain the financial and mathematical ground to the problem, we reviewed several papers on Portfolio Theory   to understand Markowitz Mean-Variance optimization analysis for a single time period. 

Given, the single-period returns for m-variate random vector R = [R1, R2, R3.., Rm]’ and the mean (expected payoffs) and variance-covariance matrix for our m-variate returns:


We denote portfolio as a weight vector of size m indicating the fraction of the total wealth held in each asset:

In such a setup, 
Total portfolio return: Rw= w’R =  , with mean portfolio return and risk as following:


Based on the theory, we simulated our Risk-adjusted returns at time t as:
Rt =w'- w'w, where  is a random variable uniformly distributed in [0, 1).

While deciding constraints to the problem, we initially stumbled upon the following decisions:
Restricting investment of complete funds before the terminal time. But, it turned out to be more of a scheduling problem rather than an optimization problem and we relaxed this constraint to make it like a self-financing scenario wherein portfolio is shuffled across assets across times.
Including a transaction cost for changing the proportion of assets was important to the project so that we could mimic the real-world scenario where a change in the proportion of asset allocation incurs a fee.

Reviewing the concepts on discrete Markovian Decision Processes as presented in Professor’s lecture notes helped us express our problem using discrete action and state spaces for the agent to choose from. 

### Methodology
#### Dataset generation:

We have considered 4 different assets for our portfolio reflective of different market instruments. The expected returns are evenly spaced values between 10 to 50, with 10 being the asset that have the least return while 50 being stock with the highest return. Similarly, asset risks are also chosen uniformly distributed between 0 and 10, with 0 denoting the choice for our risk-free asset, while 10 is the risk associated with equities, our volatile asset.

We are optimizing the investment decisions over 100 time units and the below graph plots the simulated performance of different assets with respect to time.


Based on our simulation, we obtained the following Variance-Covariance matrix:


Figure 2: Variance- Covariance of Simulated Data

We have used this Variance-Covariance matrix going forward to compute our risk adjusted returns, which serves as the utility function for our reinforcement problem.
#### Design Constraints

Our design constraint includes using only reinforcement learning to build an agent that allocates money among the assets of varying risk levels. At time t=0, the agent invests all the money he has in his hand across different assets in the portfolio. To be consistent with our results, we are taking a start allocation of [0.1, 0.2, 0.3, 0.4] respectively into government bonds, cash equivalents, real estate derivatives and equities. The option was chosen based on how a sub-optimal agent/investor would act, based on proportions of expected payoffs. We are considering one investment cycle to be of 100 time periods.

Also, we are operating in a self-financed portfolio environment wherein there is no exogenous infusion of wealth or withdrawal, and that a shuffle of allocations within the portfolio is financed by the sale of others.
There is no change to expected payoff as well as risk based on the actions that we take.
Markovian Decision Processes:

As the allocations within a portfolio is continuous over the interval [0, 1], we attempted to discretize the choices in which our portfolio can be in. We therefore chose the discrete set of [0.1, 0.2, 0.3, 0.4] as our available choices to simplify our optimization problem. Further, we restricted that the allocations can be from one of the permutations of the set. Therefore, an allocation of [0.3, 0.3, 0.2, 0.2] is not a choice that an agent can make in the RL environment.

Based on the restrictions cited above, we define the following components to our Markovian Decision Processes:

State Space
Proportion of wealth allocated to different assets 
	  {w0,w1,w2,w3}
	Where w0 = Proportion of wealth in asset 1 
w1 = Proportion of wealth in asset 2
w2 = Proportion of wealth in asset 3
w3 = Proportion of wealth in asset 4

We therefore have 24 different states in our problem, which are 24 different permutations of the set [0.1, 0.2, 0.3, 0.4]

Action Space
Set of proportions the agent can choose from to invest in. 

Again, as we have 24 different permutations to the set [0.1, 0.2, 0.3, 0.4], we have 24 actions to choose from at each time step.

Rewards
The returns obtained after choosing an action.

Rp = R + (1 − w’1m)r0

where 
Rp = single time portfolio reward where p represents portfolio
r0 = return for risk-free asset
(1 − w’1m) = weight for the risk-free asset
w = weights allocated across risky asset and w’ is the transpose

R = w’α − λw’Σw, is the risk adjusted returns at time t where
α = payoff for risky
Σ = variance-covariance for our portfolio
Λ = standard normal random variable

Further, we have a transaction cost with which the agent is penalized when it seeks to shuffle across the portfolio from the current state. 

Therefore, we define our Reward Function in terms of current state w and action a (the next state being result of our action) as:

r(s, a) = Rp + || s - a || * Transaction_fee * Balance Invested, 
where, 
Rp =  Risk adjusted returns as described above and is a function of actions a.
Transaction_fee = we are considering a transaction fee of 1% for our problem.
Balance Invested = total balance invested in the portfolio at t=0.

We seek to optimize the cumulative rewards earned over the investment cycle (total time T).

Discount Factor

As the future value of money is not the same as if the same amount is earned today, but rather lower due to several factors like inflation or other macroeconomic conditions, therefore we discount the rewards earned in future time steps.

Therefore, our discounted cumulative rewards:

G = t = 0T -1tRt+1


For our problem, we have decided on a discount factor of = 0.9. Refer to Challenges on how we came up with the value for discount factor.

The Q-value which quantifies the attractiveness of a state has a recursive relation and is given by the following Bellman optimality equation -


We set Q values to all zero initially. Our agent attempts to find the optimal path using - greedy Q learning algorithm, where our agent randomly chooses an action with probability and with 
1- probability takes an action optimally based on maximizing the state-action values. 

For the - greedy Q-learning, we have followed a tapering schedule for over a total of 90000 episodes with   decaying every 2000 episodes. 
Following plot demonstrates the tapering schedule for our Reinforcement Learning problem:


### Tools
We have majorly used Google Colab as an interactive and collaborative Python notebook, which also provided us GPU’s to carry out compute-intensive update operations on our Q table.
Please find the source code in GitHub repository at below link - https://github.com/Shikhas/Portfolio-Management-using-RL
Results
We obtained the following results as a result of optimization to our problem - 
In order to validate if the agent has explored the optimal policy, we compared the discounted cumulative reward obtained following the optimal policy with that of 5 other random policies. Please find the below graph:

Figure 3: Comparison of Rewards from Optimal policy and 5 different Suboptimal policy


We can notice in the above plot that the cumulative returns obtained from the optimal policy tend to be higher at the terminal time than the cumulative returns obtained using five other sub-optimal random policies, which aligns with our expected results.

Below plot represents rewards over the episodes of time period:

 
Figure 4: Cumulative rewards with respect to number of episodes

The graph depicts how the variation in cumulative returns changes as the number of episode increases. While the agent has successfully converged over an optimal policy over episodes, there seems to be episodes early on demonstrating a higher return over the investment cycle. The same has also been included in Limitations to Results.

We plotted Optimal Path cumulative rewards against cumulative rewards obtained investing  single asset .


Figure 5: Returns of different assets v/s  Optimal policy Returns


The graph suggests that the rewards obtained with individual investments on asset such as investing only on suppose Government Bond (indicated by orange line in the above plot) does not yield as much rewards as diversified portfolio cumulative reward obtained by investing different proportions in mix of four different kinds of assets of Government Bonds, Real Estate derivatives, Equities and Cash Equivalents.

	
Density plots for various asset allocation for four quarters- 
First Quarter( uptill  25 units of time), 
Second Quarter( 25 to 50 units of time), 
Third Quarter ( 50 to 75 units of time), 
Fourth Quarter( 75 to 100 units of time)

We can infer from below four density plots that in the first quarter, optimal policy invests a higher proportion in Cash Equivalents asset. Then moving to second quarter, higher proportion shifted to Real Estate derivatives and in the third quarter, investments are more in both Cash Equivalents and Real Estate derivatives. While, in the final quarter the higher proportion of investments are noted in Government Bonds.
 

Figure 6: Density plot of Assets in 1st Quarter 

Figure 7: Density plot of Assets in 2nd Quarter

Figure 8: Density plot of Assets in 3rd Quarter

Figure 9: Density plot of Assets in 4th Quarter

Above graphs explains how algorithm shuffles our funds in different assets in different quarters and identifies the optimal proportion that maximises the reward at the end of investment time.
### Challenges
As in our problem, the number of states and actions the agent can be in is 24. This leaves us with 57600 (24 x 100 x 24) cells to be updated over each episode considering a 100 time period as one episode.



Key Challenges faced during the project implementation were 
To link our simulated data to reflect real market scenarios, we faced challenges in identifying the distribution for different assets. As we didn’t have a real historical order book data, we simulated based on values randomly selected out of hindsights as described in the report.
Identifying the total number of episodes required for the agent to learn the optimal path.  Different number of episodes were tried so as to reach an optimal policy. We tried using 2000, 20000, 25000, 30000 number of episodes so as to check if the agent, reached the optimal path. But, as the updates per episode were 57600, we considered 90000 number of episodes to be the best choice given the agent explores most of the suboptimal paths running that longer.
The optimal discount factor for Q-learning. To capture the nature of inflation we decided to keep the discounting factor to 0.9. The future rewards were discounted 10% over their value. In case discount factor is too low, we might end up not investing but spend more. After some discussion we decided to keep the discount factor to 0.9 to mimic the real-world scenario.
Identifying how to obtain a proper calculation for transaction cost to be applied. A transaction cost of 1% was claimed on Balance amount when there was a change in state at time t+1 from time t.
Identifying the optimal step size and epsilon for greedy exploration policy. 
Given the number of updates per episode, policy iteration would have taken large resources for the agent to reach an optimal path. Hence in order for the faster implementation of the project code, we decided to go for value-iteration instead of policy iteration.
### Limitations to Results

Figure 4 i.e cumulative rewards with respect to number of episodes suggests that some earlier episodes had a higher cumulative reward than the reward at which our optimal policy converged. We are unable to identify the reason behind this behaviour.

We would also like to consider different representations of our state-action value functions to increase the portion of the state-action space which are not zeros (not explored by our algorithm). Doing so would help us increase the coverage and accuracy to our problem. The current tabular representation considered in our problem though provides an accurate and precise representation for MDPs, but often fails to generalize thus impacting the learning time to reach optimal policies.
Conclusion
In this project we tried to understand how an agent tries to maximize his profits, when he is provided with choices of different assets with varying risk and returns. Initial design of project aimed at providing the agent a constraint on not investing his entire funds into different assets until the end of the investment cycle. This restricted our project to be a scheduling project. Since we were keen on learning how Reinforcement Learning can be used in asset allocation problem, we decided to allow the agent to invest his entire balance in the portfolio to mimic the self-financing scenario. We also learnt that diversifying the allocation hedges the risk associated with each asset.

For such a scenario, we found out the optimal allocation of money to mixture of assets containing risk free assets as well as risky assets by using the Q-Learning algorithm. This will help any person who has decided to invest a fixed amount of money for T units of period for asset allocation to achieve maximum return with intrinsically minimizing risk by allocating his/her money to different kinds of assets which include Government Bond, Cash Equivalent, Real Estate derivatives and Equities.

### Further Steps

We would like to incorporate continuous state and action spaces to our problem to make it a real-market scenario.
Incorporate different representations to state-action value space to generalize the model better.
We are running the algorithm on synthetically generated data. The return rates and the risk is set based on our understanding of the market. As part of further work, this algorithm can be run on real data and checked if the algorithm provides with correct results.
Only four different types of assets were considered in our project but depending upon the investor’s objective, we can consider different types of assets such as agriculture, livestock, money market vehicles, etc. Also, analysis can be extended by increasing the number of assets to be considered (more than 4 assets as considered in our project). 

