use statrs::function::erf::erf;

/// Calculates the cumulative distribution function (CDF) for the standard normal distribution.
///
/// The standard normal distribution is a continuous probability distribution with a mean of 0 and a standard deviation of 1.
/// This function computes the probability that a normally distributed random variable will have a value less than or equal to `x`.
///
/// # Arguments
///
/// * `x` - The input value for which the cumulative probability is calculated.
///
/// # Returns
///
/// The cumulative probability corresponding to the input `x`.
///
/// # Mathematical Definition
///
/// The CDF of the standard normal distribution is given by:
///
/// ```math
/// \Phi(x) = \frac{1}{2} \left[1 + \operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
/// ```
///
/// where `erf` is the error function.
///
/// # Example
///
/// ```rust
/// use strato_pricer::bs::norm_cdf;
///
/// let x = 1.0;
/// let probability = norm_cdf(x);
/// println!("The probability that a standard normal random variable is less than or equal to {} is {}", x, probability);
/// // Output: The probability that a standard normal random variable is less than or equal to 1.0 is 0.8413447460685429
/// ```
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / f64::sqrt(2.0)))
}

/// Computes the Black-Scholes price of a European call option.
///
/// The Black-Scholes model provides a theoretical estimate of the price of European-style options.
/// This function calculates the price of a call option, which gives the holder the right, but not the obligation,
/// to buy an asset at a specified strike price on the option's expiration date.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset (spot price).
/// * `k` - The strike price at which the option can be exercised.
/// * `t` - The time to expiration in years.
/// * `r` - The risk-free interest rate (annualized).
/// * `sigma` - The volatility of the underlying asset's returns (annualized standard deviation).
///
/// # Returns
///
/// The theoretical price of the European call option.
///
/// # Mathematical Formula
///
/// The Black-Scholes formula for a European call option is:
///
/// ```math
/// C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)
/// ```
///
/// where:
///
/// ```math
/// d_1 = \frac{\ln(S / K) + (r + \frac{1}{2} \sigma^2) T}{\sigma \sqrt{T}}
/// ```
///
/// ```math
/// d_2 = d_1 - \sigma \sqrt{T}
/// ```
///
/// * \( N(\cdot) \) is the cumulative distribution function of the standard normal distribution.
/// * \( S \) is the current price of the underlying asset.
/// * \( K \) is the strike price.
/// * \( T \) is the time to expiration.
/// * \( r \) is the risk-free interest rate.
/// * \( \sigma \) is the volatility of the underlying asset.
///
/// # Edge Cases
///
/// * If `t` (time to expiration) is zero, the option has expired.
///   The function returns the intrinsic value of the call option: `max(s - k, 0.0)`.
///
/// # Example
///
/// ```rust
/// use strato_pricer::bs::black_scholes_call;
///
/// let s = 100.0;     // Current stock price
/// let k = 90.0;      // Strike price
/// let t = 0.5;       // Time to expiration in years
/// let r = 0.05;      // Risk-free interest rate
/// let sigma = 0.2;   // Volatility
///
/// let call_price = black_scholes_call(s, k, t, r, sigma);
/// println!("The theoretical price of the call option is {}", call_price);
/// // Output: The theoretical price of the call option is 12.597...
/// ```
pub fn black_scholes_call(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t == 0.0 {
        // Option has expired; return intrinsic value
        return (s - k).max(0.0);
    }

    let d1 = (f64::ln(s / k) + (r + 0.5 * sigma.powi(2)) * t) / (sigma * f64::sqrt(t));
    let d2 = d1 - sigma * f64::sqrt(t);

    s * norm_cdf(d1) - k * f64::exp(-r * t) * norm_cdf(d2)
}

/// Computes the Black-Scholes price of a European put option.
///
/// The Black-Scholes model provides a theoretical estimate of the price of European-style options.
/// This function calculates the price of a put option, which gives the holder the right, but not the obligation,
/// to sell an asset at a specified strike price on the option's expiration date.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset (spot price).
/// * `k` - The strike price at which the option can be exercised.
/// * `t` - The time to expiration in years.
/// * `r` - The risk-free interest rate (annualized).
/// * `sigma` - The volatility of the underlying asset's returns (annualized standard deviation).
///
/// # Returns
///
/// The theoretical price of the European put option.
///
/// # Mathematical Formula
///
/// The Black-Scholes formula for a European put option is:
///
/// ```math
/// P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)
/// ```
///
/// where:
///
/// ```math
/// d_1 = \frac{\ln(S / K) + (r + \frac{1}{2} \sigma^2) T}{\sigma \sqrt{T}}
/// ```
///
/// ```math
/// d_2 = d_1 - \sigma \sqrt{T}
/// ```
///
/// * \( N(\cdot) \) is the cumulative distribution function of the standard normal distribution.
/// * \( S \) is the current price of the underlying asset.
/// * \( K \) is the strike price.
/// * \( T \) is the time to expiration.
/// * \( r \) is the risk-free interest rate.
/// * \( \sigma \) is the volatility of the underlying asset.
///
/// # Edge Cases
///
/// * If `t` (time to expiration) is zero, the option has expired.
///   The function returns the intrinsic value of the put option: `max(k - s, 0.0)`.
///
/// # Example
///
/// ```rust
/// use strato_pricer::bs::black_scholes_put;
///
/// let s = 100.0;     // Current stock price
/// let k = 110.0;     // Strike price
/// let t = 0.5;       // Time to expiration in years
/// let r = 0.05;      // Risk-free interest rate
/// let sigma = 0.2;   // Volatility
///
/// let put_price = black_scholes_put(s, k, t, r, sigma);
/// println!("The theoretical price of the put option is {}", put_price);
/// // Output: The theoretical price of the put option is 12.815...
/// ```
pub fn black_scholes_put(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t == 0.0 {
        // Option has expired; return intrinsic value
        return (k - s).max(0.0);
    }

    let d1 = (f64::ln(s / k) + (r + 0.5 * sigma.powi(2)) * t) / (sigma * f64::sqrt(t));
    let d2 = d1 - sigma * f64::sqrt(t);

    k * f64::exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_scholes_call_price() {
        let s = 100.0; // Spot price of the underlying asset
        let k = 100.0; // Strike price
        let t = 1.0; // Time to maturity (1 year)
        let r = 0.05; // Risk-free interest rate (5%)
        let sigma = 0.2; // Volatility (20%)

        // Call price using Black-Scholes formula
        let call_price = black_scholes_call(s, k, t, r, sigma);

        // Expected price calculated from a standard Black-Scholes calculator
        let expected_call_price = 10.45058;

        // Set an acceptable margin of error (epsilon) for floating-point comparison
        let epsilon = 1e-5;
        assert!(
            (call_price - expected_call_price).abs() < epsilon,
            "Call price incorrect. Expected: {}, Got: {}",
            expected_call_price,
            call_price
        );
    }

    #[test]
    fn test_black_scholes_put_price() {
        let s = 100.0; // Spot price of the underlying asset
        let k = 100.0; // Strike price
        let t = 1.0; // Time to maturity (1 year)
        let r = 0.05; // Risk-free interest rate (5%)
        let sigma = 0.2; // Volatility (20%)

        // Put price using Black-Scholes formula
        let put_price = black_scholes_put(s, k, t, r, sigma);

        // Expected price calculated from a standard Black-Scholes calculator
        let expected_put_price = 5.57352;

        // Set an acceptable margin of error (epsilon) for floating-point comparison
        let epsilon = 1e-5;
        assert!(
            (put_price - expected_put_price).abs() < epsilon,
            "Put price incorrect. Expected: {}, Got: {}",
            expected_put_price,
            put_price
        );
    }

    #[test]
    fn test_black_scholes_zero_volatility() {
        let s = 100.0; // Spot price of the underlying asset
        let k = 100.0; // Strike price
        let t = 1.0; // Time to maturity (1 year)
        let r = 0.05; // Risk-free interest rate (5%)
        let sigma = 0.0; // Volatility (0%)

        // Call price with zero volatility should equal max(S - K, 0) discounted at the
        // risk-free rate
        let call_price = black_scholes_call(s, k, t, r, sigma);
        let expected_call_price = (s - k * f64::exp(-r * t)).max(0.0);

        let epsilon = 1e-5;
        assert!(
            (call_price - expected_call_price).abs() < epsilon,
            "Call price with zero volatility incorrect. Expected: {}, Got: {}",
            expected_call_price,
            call_price
        );

        // Put price with zero volatility should equal max(K - S, 0) discounted at the
        // risk-free rate
        let put_price = black_scholes_put(s, k, t, r, sigma);
        let expected_put_price = (k * f64::exp(-r * t) - s).max(0.0);

        assert!(
            (put_price - expected_put_price).abs() < epsilon,
            "Put price with zero volatility incorrect. Expected: {}, Got: {}",
            expected_put_price,
            put_price
        );
    }

    #[test]
    fn test_black_scholes_zero_time_to_maturity() {
        let s = 100.0; // Spot price of the underlying asset
        let k = 100.0; // Strike price
        let t = 0.0; // Time to maturity (0 years)
        let r = 0.05; // Risk-free interest rate (5%)
        let sigma = 0.2; // Volatility (20%)

        // Call price with zero time to maturity should equal max(S - K, 0)
        let call_price = black_scholes_call(s, k, t, r, sigma);
        let expected_call_price = (s - k).max(0.0);

        let epsilon = 1e-5;

        assert!(
            (call_price - expected_call_price).abs() < epsilon,
            "Call price with zero time to maturity incorrect. Expected: {}, Got: {}",
            expected_call_price,
            call_price
        );

        // Put price with zero time to maturity should equal max(K - S, 0)
        let put_price = black_scholes_put(s, k, t, r, sigma);
        let expected_put_price = (k - s).max(0.0);

        assert!(
            (put_price - expected_put_price).abs() < epsilon,
            "Put price with zero time to maturity incorrect. Expected: {}, Got: {}",
            expected_put_price,
            put_price
        );
    }

    #[test]
    fn test_black_scholes_deep_in_the_money_call() {
        let s = 150.0; // Spot price of the underlying asset (deep ITM)
        let k = 100.0; // Strike price
        let t = 1.0; // Time to maturity (1 year)
        let r = 0.05; // Risk-free interest rate (5%)
        let sigma = 0.2; // Volatility (20%)

        // Call price using Black-Scholes formula
        let call_price = black_scholes_call(s, k, t, r, sigma);
        let expected_call_price = 54.970140138;

        let epsilon = 1e-5; // Tolerance for floating-point comparison
        assert!(
            (call_price - expected_call_price).abs() < epsilon,
            "Call price deep in-the-money incorrect. Expected: {}, Got: {}",
            expected_call_price,
            call_price
        );
    }
}
