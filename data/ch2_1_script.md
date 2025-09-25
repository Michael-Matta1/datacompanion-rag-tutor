A hypothesis is a statement introducing a possible explanation for an event. We use it as a starting point for validating a larger effort. A strong hypothesis must be tested using data, declared as a statement, not a question, and concise enough to break the work into manageable pieces that follow sound logic. 
This enables us to iterate systematically, confirm our understanding and generalize beyond our sample data, and develop actionable recommendations.

The simple format of a hypothesis is an if then statement. In AB testing, a stronger format is desirable. It follows: based on X, if we do Y, then Z will happen as measured by metric M. For example, based on user experience research, we believe that if we update our checkout page design, the percentage of purchasing customers will increase as measured by purchase rate. This is referred to as the alternative hypothesis. The null hypothesis assumes no change.

To test this hypothesis, we first calculate the purchase rates in our default checkout page design A and updated design B. Examining the checkout dataset, let's count the number of users in each group using the count method and calculate the mean purchase rate using the mean method. We see that each group has 3000 users and the purchase rate in group B is slightly higher than A.

Recall that the number of purchasers in n trials with purchasing probability p is binomially distributed.
To plot the distributions, we import binom from scipy, use the binom-dot-pmf method, which takes the x-axis range, the number of trials, and the probability of success as arguments, and create a bar plot using matplotlib. These distributions tell us the probabilities of, at most, a certain number of users purchasing in each group, but they don't tell us anything about the long-term purchase probability of the overall population if we were to present either checkout page to the users. To do so, we need to make inferences on the means using a normal distribution.

Recall that the central limit theorem states that as the sample size gets larger, the distribution of the sample means, p, will be normally distributed around the true population mean with a standard deviation equal to the standard error of the mean irrespective of the shape of the distribution of the data. We use this to approximate the true mean of the population from which the data was sampled using this formula where p and n are the mean proportion and the sample size respectively.

To demonstrate with Python, we set a random seed, create an empty list to hold the sampled means, and create a loop to simulate a thousand means with replacement for sample sizes 10, 25, and 100. Plotting the distributions using Seaborn's displot method shows that as the sample size gets larger, the distribution of the means approaches a normal one.

Since the sample means will follow a normal distribution, we can plot them for groups A and B using scipy's norm-dot-pdf and Seaborn's lineplot using the formulas of the mean and standard error shown previously. 

The dashed lines represent the mean purchase rate for each group and the distance d is the mean difference between them. Since both distributions are normal, d also follows a normal distribution.

This allows us to represent our original hypothesis mathematically as: The null hypothesis stating that the updated checkout page will result in no difference in purchase rates between the groups, and an alternative hypothesis stating that the updated design will result in a non-zero difference in purchase rate.

