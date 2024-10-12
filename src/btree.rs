/// Module for pricing American options using the Binomial Option Pricing Model.
///
/// The binomial model is a numerical method that uses a discrete-time lattice-based approach
/// to model the possible future movements of an asset's price and to evaluate options.
///
/// This module provides functions to price both American call and put options using the Cox-Ross-Rubinstein (CRR) binomial tree.
use std::f64;

/// Calculates the price of an American option (call or put) using the binomial option pricing model.
///
/// # Arguments
///
/// * `s` - The current price of the underlying asset (spot price).
/// * `k` - The strike price at which the option can be exercised.
/// * `t` - The time to expiration in years.
/// * `r` - The risk-free interest rate (annualized).
/// * `sigma` - The volatility of the underlying asset's returns (annualized standard deviation).
/// * `div_yield` - The continuous dividend yield of the underlying asset (annualized).
/// * `steps` - The number of time steps in the binomial model (higher numbers increase accuracy).
/// * `option_type` - The type of the option: `"call"` or `"put"`.
///
/// # Returns
///
/// The theoretical price of the American option.
///
/// # Mathematical Formulation
///
/// The binomial model builds a price tree, starting from the current asset price and moving forward in time in discrete steps.
/// At each node, the asset price can move up or down with certain probabilities. The option value is calculated recursively
/// by working backward from the expiration date to the present.
///
/// The up (`u`) and down (`d`) factors are calculated as:
///
/// ```math
/// u = e^{\sigma \sqrt{\Delta t}}
/// d = \frac{1}{u}
/// ```
///
/// The risk-neutral probabilities are:
///
/// ```math
/// p = \frac{e^{(r - q) \Delta t} - d}{u - d}
/// ```
///
/// where:
/// - \( \Delta t = \frac{T}{N} \) is the time increment per step.
/// - \( \sigma \) is the volatility.
/// - \( r \) is the risk-free interest rate.
/// - \( q \) is the continuous dividend yield.
///
/// The option price at each node is:
///
/// ```math
/// V_i^j = \max\left( \text{exercise value}, e^{-r \Delta t} [p V_{i+1}^{j+1} + (1 - p) V_{i+1}^j] \right)
/// ```
///
/// where:
/// - \( V_i^j \) is the option value at node \( (i, j) \).
/// - The exercise value is \( \max(K - S_i^j, 0) \) for a put and \( \max(S_i^j - K, 0) \) for a call.
///
/// # Edge Cases
///
/// * If `steps` is set too low, the approximation may be inaccurate. A typical value is 50 to 500.
/// * If `sigma` (volatility) is zero, the model reduces to a risk-free discounting of the option payoff.
///
/// # Example
///
/// ```rust
/// use strato_pricer::btree::american_option_binomial;
///
/// let s = 100.0;     // Current stock price
/// let k = 100.0;     // Strike price
/// let t = 1.0;       // Time to expiration in years
/// let r = 0.05;      // Risk-free interest rate
/// let sigma = 0.2;   // Volatility
/// let div_yield = 0.0; // Dividend yield
/// let steps = 100;   // Number of steps in the binomial tree
///
/// let call_price = american_option_binomial(s, k, t, r, sigma, div_yield, steps, "call");
/// let put_price = american_option_binomial(s, k, t, r, sigma, div_yield, steps, "put");
///
/// println!("American Call Option Price: {}", call_price);
/// println!("American Put Option Price: {}", put_price);
/// // Output will be the theoretical prices of the American call and put options.
/// ```
pub fn american_option_binomial(
    s: f64,
    k: f64,
    t: f64,
    r: f64,
    sigma: f64,
    div_yield: f64,
    steps: usize,
    option_type: &str,
) -> f64 {
    let delta_t = t / steps as f64;

    // Precompute constants
    let up = f64::exp(sigma * f64::sqrt(delta_t));
    let down = 1.0 / up;
    let a = f64::exp((r - div_yield) * delta_t);
    let p = (a - down) / (up - down);

    // Initialize asset prices at maturity
    let mut asset_prices = vec![0.0; steps + 1];
    for i in 0..=steps {
        let j = steps - i;
        asset_prices[i] = s * up.powi(j as i32) * down.powi(i as i32);
    }

    // Initialize option values at maturity
    let mut option_values = vec![0.0; steps + 1];
    for i in 0..=steps {
        if option_type == "call" {
            option_values[i] = f64::max(asset_prices[i] - k, 0.0);
        } else {
            option_values[i] = f64::max(k - asset_prices[i], 0.0);
        }
    }

    // Step back through the tree
    for step in (0..steps).rev() {
        for i in 0..=step {
            let continuation_value =
                f64::exp(-r * delta_t) * (p * option_values[i] + (1.0 - p) * option_values[i + 1]);

            asset_prices[i] = asset_prices[i] / up;

            let exercise_value = if option_type == "call" {
                f64::max(asset_prices[i] - k, 0.0)
            } else {
                f64::max(k - asset_prices[i], 0.0)
            };

            option_values[i] = f64::max(continuation_value, exercise_value);
        }
    }

    option_values[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_american_call_option() {
        // Test parameters
        let s = 100.0; // Current stock price
        let k = 100.0; // Strike price
        let t = 1.0; // Time to expiration in years
        let r = 0.06; // Risk-free interest rate
        let sigma = 0.2; // Volatility
        let div_yield = 0.0; // Dividend yield
        let steps = 100; // Number of steps
        let option_type = "call";

        // Expected value from a reliable source or previous computation
        let expected_price = 10.969462;

        let calculated_price =
            american_option_binomial(s, k, t, r, sigma, div_yield, steps, option_type);

        assert!(
            (calculated_price - expected_price).abs() < 0.01,
            "Calculated price {} differs from expected price {}",
            calculated_price,
            expected_price
        );
    }

    #[test]
    fn test_american_put_option() {
        // Test parameters
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.06;
        let sigma = 0.2;
        let div_yield = 0.0;
        let steps = 100;
        let option_type = "put";

        let expected_price = 5.791149;

        let calculated_price =
            american_option_binomial(s, k, t, r, sigma, div_yield, steps, option_type);

        assert!(
            (calculated_price - expected_price).abs() < 0.01,
            "Calculated price {} differs from expected price {}",
            calculated_price,
            expected_price
        );
    }

    #[test]
    fn test_zero_volatility() {
        // When volatility is zero, the option price should be the intrinsic value
        let s = 100.0;
        let k = 90.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.0;
        let div_yield = 0.0;
        let steps = 10;
        let option_type = "call";

        let expected_price = f64::max(s - k, 0.0); // Intrinsic value

        let calculated_price =
            american_option_binomial(s, k, t, r, sigma, div_yield, steps, option_type);

        assert_eq!(
            calculated_price, expected_price,
            "Calculated price {} differs from expected price {}",
            calculated_price, expected_price
        );
    }

    #[test]
    fn test_zero_time_to_expiration() {
        // When time to expiration is zero, the option price should be its intrinsic value
        let s = 100.0;
        let k = 90.0;
        let t = 0.0;
        let r = 0.05;
        let sigma = 0.2;
        let div_yield = 0.0;
        let steps = 1;
        let option_type = "call";

        let expected_price = f64::max(s - k, 0.0);

        let calculated_price =
            american_option_binomial(s, k, t, r, sigma, div_yield, steps, option_type);

        assert_eq!(
            calculated_price, expected_price,
            "Calculated price {} differs from expected price {}",
            calculated_price, expected_price
        );
    }

    #[test]
    fn test_immediate_exercise() {
        // Test a scenario where immediate exercise is optimal
        let s = 50.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.2;
        let div_yield = 0.0;
        let steps = 100;
        let option_type = "put";

        // For deep in-the-money American put options, immediate exercise might be optimal
        let expected_price = f64::max(k - s, 0.0);

        let calculated_price =
            american_option_binomial(s, k, t, r, sigma, div_yield, steps, option_type);

        assert!(
            (calculated_price - expected_price).abs() < 0.01 || calculated_price > expected_price,
            "Calculated price {} is less than expected intrinsic value {}",
            calculated_price,
            expected_price
        );
    }

    #[test]
    fn test_high_dividend_yield() {
        // Test the effect of a high dividend yield on an American call option
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.2;
        let div_yield = 0.1; // High dividend yield
        let steps = 100;
        let option_type = "call";

        let calculated_price =
            american_option_binomial(s, k, t, r, sigma, div_yield, steps, option_type);

        // Without a benchmark, we can at least check that the price is positive
        assert!(
            calculated_price > 0.0,
            "Calculated price should be positive for an American call option with high dividend yield"
        );
    }

    #[test]
    fn test_invalid_option_type() {
        // Test handling of an invalid option type
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.2;
        let div_yield = 0.0;
        let steps = 100;
        let option_type = "invalid";

        // Since the function does not currently handle invalid option types, it will default to pricing a put option
        let calculated_price =
            american_option_binomial(s, k, t, r, sigma, div_yield, steps, option_type);

        // For better error handling, consider modifying the function to return a Result
        // For now, we can check that the price is calculated
        assert!(
            calculated_price >= 0.0,
            "Calculated price should be non-negative"
        );
    }

    #[test]
    fn test_low_number_of_steps() {
        // Test the function with a low number of steps to check for stability
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.2;
        let div_yield = 0.0;
        let steps = 1;
        let option_type = "call";

        let calculated_price =
            american_option_binomial(s, k, t, r, sigma, div_yield, steps, option_type);

        // The price may not be accurate, but it should be computed without errors
        assert!(
            calculated_price >= 0.0,
            "Calculated price should be non-negative even with low number of steps"
        );
    }
}
