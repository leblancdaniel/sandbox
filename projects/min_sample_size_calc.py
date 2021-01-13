import scipy.stats as scs


# determine minimum sample size needed in test given mean, difference, power and alpha
def min_sample_size(m1, mde, power=0.8, alpha=0.05):
    """
    Returns the minimum sample size to set up a split test

    Arguments:
        m1 (float): probability of success for control
        mde (float): minimum change in measurement between control
        group and test group if alternative hypothesis is true, sometimes
        referred to as minimum detectable effect
        power (float): probability of rejecting the null hypothesis when the
        null hypothesis is false, typically 0.8
        alpha (float): significance level of test, typically 0.05
    Returns:
        min_N: minimum sample size of each group (float)
    References:
        Stanford lecture on sample sizes
        http://statweb.stanford.edu/~susan/courses/s141/hopower.pdf
    """

    # standard normal distribution to determine z-values
    standard_norm = scs.norm(0, 1)

    Z_beta = standard_norm.ppf(power)
    print("Z_beta (power):", Z_beta)

    Z_alpha = standard_norm.ppf(1-alpha/2)
    print("Z_alpha (1 - alpha/2:", Z_alpha)

    # average of probabilities from both groups
    pooled_prob = (m1 + m1+mde) / 2

    min_N = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2
             / mde**2)

    print("""
        With a {}% significance level, 
        we need a Minimum Sample Size of {:.2f} per group 
        to have a {}% chance of 
        correctly detecting a difference of {:.2f}.
        """.format(alpha*100, min_N, power*100, mde))

min_sample_size(m1=0.10, mde=0.02, power=0.9, alpha=0.01)


# determine power of test given sample size, mean of test and control groups, and alpha
def get_power(n, m1, m2, alpha=0.05):
    """
    Returns statistical power, given a A/B test sample size, group means, and alpha.
    Bigger sample size -> bigger power
    Bigger alpha -> bigger power

    Arguments:
        n (int):        sample size of each group in A/B test
        m1 (float):     mean of control group
        m2 (float):     mean of test group
        alpha (float):  significance level of test, typically 0.05
    Returns:
        power:          statistical power of test
    """

    Z_alpha = scs.norm.ppf(1 - alpha/2)
    mde = abs(m2-m1)
    bp = (m1+m2) / 2
    
    v1 = m1 * (1-m1)
    v2 = m2 * (1-m2)
    bv = bp * (1-bp)
    
    power_part_one = scs.norm.cdf((n**0.5 * mde - Z_alpha * (2 * bv)**0.5) / (v1+v2) ** 0.5)
    power_part_two = 1 - scs.norm.cdf((n**0.5 * mde + Z_alpha * (2 * bv)**0.5) / (v1+v2) ** 0.5)
    
    power = power_part_one + power_part_two
    
    print("""
        With a {}% significance level and
        given a sample size of {},
        we have a {:.2f}% chance,
        of correctly detecting a difference of {:.2f}.
        """.format(alpha*100, n, power*100, mde))

get_power(n=7283, m1=0.10, m2=0.12, alpha=0.01)
get_power(n=7283, m1=0.10, m2=0.12, alpha=0.05)


# get p-value of test results given mean and sample size of control and test groups
def get_pvalue(m1, m2, n1, n2):
    """
    Returns the p-value of the result of mean difference between two groups

    Arguments:
        m1 (float):     mean of control group
        m2 (float):     mean of test group
        n1 (int):       sample size of control group 
        n2 (int):       sample size of test group
    Returns:
        p_value (float): probability of getting observed results if truly no difference between groups
    """

    mde = -abs(m2 - m1)
    
    scale_one = m1 * (1 - m1) * (1 / n1)
    scale_two = m2 * (1 - m2) * (1 / n2)
    scale_val = (scale_one + scale_two) ** 0.5
    
    p_value = 2 * scs.norm.cdf(mde, loc=0, scale=scale_val)  
    
    print("""
        Group 1 N = {}, mean = {}
        Group 2 N = {}, mean = {}
        p-value = {}
        """.format(n1, m1, n2, m2, p_value))

get_pvalue(0.10, 0.12, 2000, 2000)


# determine confidence interval of difference of means between groups
def get_conf_interval(m1, m2, n1, n2, alpha=0.05):
    """
    Returns the confidence interval of difference of means, given their sample sizes

    Arguments:
        m1 (float):     mean of control group
        m2 (float):     mean of test group
        n1 (int):       sample size of control group
        n2 (int):       sample size of test group
        alpha (float):  significance level of test, usually 0.05
    Returns:
        Confidence Interval of difference of mean at significance level, alpha
    """

    mde = m2 - m1
    var = (1 - m2) * m2 / n2 + (1 - m1) * m1 / n1
    stdev = var**0.5

    val = abs(scs.norm.ppf(alpha/2))
    lower_bound = mde - val * stdev
    upper_bound = mde + val * stdev

    print("""
        We can be {:.2f}% confident that 
        the true mean difference is between {} and {}.
        """.format((1-alpha)*100, lower_bound, upper_bound))

get_conf_interval(0.10, 0.12, 8000, 8000)


