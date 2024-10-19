use statrs::function::erf::erf;

use crate::OptionKind;

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
/// // Output: The probability that a standard normal random variable is less than or equal to 1.0 is 0.8413...
/// ```
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / f64::sqrt(2.0)))
}

/// Computes the Black-Scholes price of a call option.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset (spot price).
/// * `k` - The strike price.
/// * `t` - The time to expiration in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility (standard deviation of the asset's returns).
///
/// # Returns
///
/// The theoretical price of the call option.
///
/// # Example
///
/// ```rust
/// use strato_pricer::bs::black_scholes_call;
///
/// let call_price = black_scholes_call(100.0, 100.0, 1.0, 0.05, 0.2);
/// println!("Call Price: {}", call_price);
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

/// Computes the Black-Scholes price of a put option.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset (spot price).
/// * `k` - The strike price.
/// * `t` - The time to expiration in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility (standard deviation of the asset's returns).
///
/// # Returns
///
/// The theoretical price of the put option.
///
/// # Example
///
/// ```rust
/// use strato_pricer::bs::black_scholes_put;
///
/// let put_price = black_scholes_put(100.0, 100.0, 1.0, 0.05, 0.2);
/// println!("Put Price: {}", put_price);
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

/// Computes the Black-Scholes price of a option (call or put).
/// 
/// # Arguments
///
/// * `s` - The current price of the underlying asset (spot price).
/// * `k` - The strike price.
/// * `t` - The time to expiration in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility (standard deviation of the asset's returns).
///
/// # Returns
///
/// The theoretical price of the put option.
///
/// # Example
/// 
/// ```rust
/// use strato_pricer::OptionKind;
/// use strato_pricer::bs::black_scholes;
/// 
/// let call_price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, OptionKind::Call);
/// assert_eq!(format!("{:.2}", call_price), "10.45");
/// 
/// let put_price = black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, OptionKind::Put);
/// assert_eq!(format!("{:.2}", put_price), "5.57");
/// ```
pub fn black_scholes(s: f64, k: f64, t: f64, r: f64, sigma: f64, option_kind: OptionKind) -> f64 {
    match option_kind {
        OptionKind::Call => black_scholes_call(s, k, t, r, sigma),
        OptionKind::Put => black_scholes_put(s, k, t, r, sigma),
    }
}

/// Computes the delta of a call option using the Black-Scholes model.
///
/// Delta represents the sensitivity of the option's price to a $1 change in the price of the underlying asset.
/// For a call option, delta ranges between 0 and 1.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset (spot price).
/// * `k` - The strike price.
/// * `t` - The time to expiration in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility (standard deviation of the asset's returns).
///
/// # Returns
///
/// The delta of the call option.
///
/// # Example
///
/// ```rust
/// use strato_pricer::bs::black_scholes_call_delta;
///
/// let delta = black_scholes_call_delta(100.0, 100.0, 1.0, 0.05, 0.2);
/// println!("Call Delta: {}", delta);
/// ```
pub fn black_scholes_call_delta(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t == 0.0 {
        // Option has expired
        if s > k {
            return 1.0;
        } else {
            return 0.0;
        }
    }

    let d1 = (f64::ln(s / k) + (r + 0.5 * sigma.powi(2)) * t) / (sigma * f64::sqrt(t));

    norm_cdf(d1)
}

/// Computes the delta of a put option using the Black-Scholes model.
///
/// Delta represents the sensitivity of the option's price to a $1 change in the price of the underlying asset.
/// For a put option, delta ranges between -1 and 0.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset (spot price).
/// * `k` - The strike price.
/// * `t` - The time to expiration in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility (standard deviation of the asset's returns).
///
/// # Returns
///
/// The delta of the put option.
///
/// # Example
///
/// ```rust
/// use strato_pricer::bs::black_scholes_put_delta;
///
/// let delta = black_scholes_put_delta(100.0, 100.0, 1.0, 0.05, 0.2);
/// println!("Put Delta: {}", delta);
/// ```
pub fn black_scholes_put_delta(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
    if t == 0.0 {
        // Option has expired
        if s < k {
            return -1.0;
        } else {
            return 0.0;
        }
    }

    let d1 = (f64::ln(s / k) + (r + 0.5 * sigma.powi(2)) * t) / (sigma * f64::sqrt(t));

    norm_cdf(d1) - 1.0
}

/// Computes the delta of a option (call or put) using the Black-Scholes model.
///
/// Delta represents the sensitivity of the option's price to a $1 change in the price of the underlying asset.
/// For a call option, delta ranges between 0 and 1. For a put option, delta ranges between -1 and 0.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset (spot price).
/// * `k` - The strike price.
/// * `t` - The time to expiration in years.
/// * `r` - The risk-free interest rate.
/// * `sigma` - The volatility (standard deviation of the asset's returns).
/// * `option_kind` - The type of option (call or put).
///
/// # Returns
///
/// The delta of the option.
///
/// # Example
///
/// ```rust
/// use strato_pricer::OptionKind;
/// use strato_pricer::bs::black_scholes_delta;
///
/// let call_delta = black_scholes_delta(100.0, 100.0, 1.0, 0.05, 0.2, OptionKind::Call);
/// assert_eq!(format!("{:.2}", call_delta), "0.64");
///
/// let put_delta = black_scholes_delta(100.0, 100.0, 1.0, 0.05, 0.2, OptionKind::Put);
/// assert_eq!(format!("{:.2}", put_delta), "-0.36");
/// ```
pub fn black_scholes_delta(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    option_kind: OptionKind,
) -> f64 {
    match option_kind {
        OptionKind::Call => black_scholes_call_delta(s, k, t, r, sigma),
        OptionKind::Put => black_scholes_put_delta(s, k, t, r, sigma),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_scholes_call_delta() {
        let s = 100.0; // Spot price
        let k = 100.0; // Strike price
        let t = 1.0; // Time to expiration
        let r = 0.05; // Risk-free rate
        let sigma = 0.2; // Volatility

        let delta = black_scholes_call_delta(s, k, t, r, sigma);
        let expected_delta = 0.6368; // Approximate value from standard tables

        let epsilon = 1e-4;
        assert!(
            (delta - expected_delta).abs() < epsilon,
            "Call delta incorrect. Expected: {}, Got: {}",
            expected_delta,
            delta
        );
    }

    #[test]
    fn test_black_scholes_put_delta() {
        let s = 100.0; // Spot price
        let k = 100.0; // Strike price
        let t = 1.0; // Time to expiration
        let r = 0.05; // Risk-free rate
        let sigma = 0.2; // Volatility

        let delta = black_scholes_put_delta(s, k, t, r, sigma);
        let expected_delta = -0.3632; // Approximate value from standard tables

        let epsilon = 1e-4;
        assert!(
            (delta - expected_delta).abs() < epsilon,
            "Put delta incorrect. Expected: {}, Got: {}",
            expected_delta,
            delta
        );
    }

    #[test]
    fn test_black_scholes_call_delta_at_expiration() {
        let s = 105.0; // Spot price greater than strike price
        let k = 100.0; // Strike price
        let t = 0.0; // Time to expiration
        let r = 0.05; // Risk-free rate
        let sigma = 0.2; // Volatility

        let delta = black_scholes_call_delta(s, k, t, r, sigma);
        let expected_delta = 1.0;

        assert_eq!(
            delta, expected_delta,
            "Call delta at expiration incorrect. Expected: {}, Got: {}",
            expected_delta, delta
        );
    }

    #[test]
    fn test_black_scholes_put_delta_at_expiration() {
        let s = 95.0; // Spot price less than strike price
        let k = 100.0; // Strike price
        let t = 0.0; // Time to expiration
        let r = 0.05; // Risk-free rate
        let sigma = 0.2; // Volatility

        let delta = black_scholes_put_delta(s, k, t, r, sigma);
        let expected_delta = -1.0;

        assert_eq!(
            delta, expected_delta,
            "Put delta at expiration incorrect. Expected: {}, Got: {}",
            expected_delta, delta
        );
    }
}
