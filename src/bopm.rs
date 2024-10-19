/// Module for pricing options using the Binomial Option Pricing Model.
///
/// The binomial model is a numerical method that uses a discrete-time lattice-based approach
/// to model the possible future movements of an asset's price and to evaluate options.
///
/// This module provides functions to price both call and put options using the Cox-Ross-Rubinstein (CRR) binomial tree.
use std::f64;

use crate::OptionKind;

pub struct Option {
    pub s: f64,
    pub k: f64,
    pub t: f64,
    pub r: f64,
    pub sigma: f64,
    pub div_yield: f64,
    pub steps: usize,
    pub kind: OptionKind,
}

/// Calculates the price of an option (call or put) using the binomial option pricing model.
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
/// * `option_kind` - The type of the option: `"call"` or `"put"`.
///
/// # Returns
///
/// The theoretical price of the option.
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
/// use strato_pricer::bopm::binomial_pricing_model;
/// use strato_pricer::bopm::Option;
/// use strato_pricer::OptionKind;
///
/// let s = 100.0;     // Current stock price
/// let k = 100.0;     // Strike price
/// let t = 1.0;       // Time to expiration in years
/// let r = 0.05;      // Risk-free interest rate
/// let sigma = 0.2;   // Volatility
/// let div_yield = 0.0; // Dividend yield
/// let steps = 100;   // Number of steps in the binomial tree
/// let call_option = OptionKind::Call;
/// let put_option = OptionKind::Put;
/// let call_option = Option {
///     s,
///     k,
///     t,
///     r,
///     sigma,
///     div_yield,
///     steps,
///     kind: call_option,
/// };
///
/// let put_option = Option {
///     s,
///     k,
///     t,
///     r,
///     sigma,
///     div_yield,
///     steps,
///     kind: put_option,
/// };
///
/// let call_price = binomial_pricing_model(call_option);
/// let put_price = binomial_pricing_model(put_option);
///
/// println!("American Call Option Price: {}", call_price);
/// println!("American Put Option Price: {}", put_price);
/// // Output will be the theoretical prices of the call and put options.
/// ```
pub fn binomial_pricing_model(option: Option) -> f64 {
    let delta_t = option.t / option.steps as f64;

    // Precompute constants
    let up = f64::exp(option.sigma * f64::sqrt(delta_t));
    let down = 1.0 / up;
    let a = f64::exp((option.r - option.div_yield) * delta_t);
    let p = (a - down) / (up - down);

    // Initialize asset prices at maturity
    let mut asset_prices = vec![0.0; option.steps + 1];
    for (i, asset_price) in asset_prices.iter_mut().enumerate().take(option.steps + 1) {
        let j = option.steps - i;
        *asset_price = option.s * up.powi(j as i32) * down.powi(i as i32);
    }

    // Initialize option values at maturity
    let mut option_values = vec![0.0; option.steps + 1];
    for i in 0..=option.steps {
        match option.kind {
            OptionKind::Call => option_values[i] = f64::max(asset_prices[i] - option.k, 0.0),
            OptionKind::Put => option_values[i] = f64::max(option.k - asset_prices[i], 0.0),
        }
    }

    // Step back through the tree
    for step in (0..option.steps).rev() {
        for i in 0..=step {
            let continuation_value = f64::exp(-option.r * delta_t)
                * (p * option_values[i] + (1.0 - p) * option_values[i + 1]);

            asset_prices[i] /= up;

            let exercise_value = match option.kind {
                OptionKind::Call => f64::max(asset_prices[i] - option.k, 0.0),
                OptionKind::Put => f64::max(option.k - asset_prices[i], 0.0),
            };

            option_values[i] = f64::max(continuation_value, exercise_value);
        }
    }

    option_values[0]
}

/// Calculates the delta of an option (call or put) using the binomial option pricing model.
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
/// * `option_kind` - The type of the option: `"call"` or `"put"`.
///
/// # Returns
///
/// The delta of the option.
///
/// # Mathematical Formulation
///
/// The delta is approximated using the option values at the first time step:
///
/// ```math
/// \Delta = \frac{V_{\text{up}} - V_{\text{down}}}{S_{\text{up}} - S_{\text{down}}}
/// ```
///
/// where:
/// - \( V_{\text{up}} \) and \( V_{\text{down}} \) are the option values after an up and down move, respectively.
/// - \( S_{\text{up}} \) and \( S_{\text{down}} \) are the asset prices after an up and down move, respectively.
///
/// # Example
///
/// ```rust
/// use strato_pricer::bopm::bopm_delta;
/// use strato_pricer::bopm::Option;
/// use strato_pricer::OptionKind;
///
/// let s = 100.0;     // Current stock price
/// let k = 100.0;     // Strike price
/// let t = 1.0;       // Time to expiration in years
/// let r = 0.05;      // Risk-free interest rate
/// let sigma = 0.2;   // Volatility
/// let div_yield = 0.0; // Dividend yield
/// let steps = 100;   // Number of steps in the binomial tree
/// let call_option = OptionKind::Call;
/// let put_option = OptionKind::Put;
/// let call_option = Option {
///     s,
///     k,
///     t,
///     r,
///     sigma,
///     div_yield,
///     steps,
///     kind: call_option,
/// };
///
/// let put_option = Option {
///     s,
///     k,
///     t,
///     r,
///     sigma,
///     div_yield,
///     steps,
///     kind: put_option,
/// };
///
/// let call_delta = bopm_delta(call_option);
/// let put_delta = bopm_delta(put_option);
///
/// println!("American Call Option Delta: {}", call_delta);
/// println!("American Put Option Delta: {}", put_delta);
/// ```
pub fn bopm_delta(option: Option) -> f64 {
    let delta_t = option.t / option.steps as f64;

    // Precompute constants
    let up = f64::exp(option.sigma * f64::sqrt(delta_t));
    let down = 1.0 / up;
    let a = f64::exp((option.r - option.div_yield) * delta_t);
    let p = (a - down) / (up - down);
    let disc = f64::exp(-option.r * delta_t);

    // Initialize asset prices and option values at maturity
    let mut asset_prices = vec![0.0; option.steps + 1];
    let mut option_values = vec![0.0; option.steps + 1];

    for i in 0..=option.steps {
        let j = option.steps - i;
        asset_prices[i] = option.s * up.powi(j as i32) * down.powi(i as i32);
        option_values[i] = match option.kind {
            OptionKind::Call => f64::max(asset_prices[i] - option.k, 0.0),
            OptionKind::Put => f64::max(option.k - asset_prices[i], 0.0),
        };
    }

    // Backward induction to find option value at t=0
    for step in (1..=option.steps).rev() {
        for i in 0..step {
            let continuation_value =
                disc * (p * option_values[i] + (1.0 - p) * option_values[i + 1]);

            asset_prices[i] /= up;

            let exercise_value = match option.kind {
                OptionKind::Call => f64::max(asset_prices[i] - option.k, 0.0),
                OptionKind::Put => f64::max(option.k - asset_prices[i], 0.0),
            };

            option_values[i] = f64::max(continuation_value, exercise_value);
        }

        // Capture the option values and asset prices at the first time step
        if step == 1 {
            // Delta calculation
            let v_up = option_values[0];
            let v_down = option_values[1];
            let s_up = asset_prices[0];
            let s_down = asset_prices[1];

            let ca = v_up - v_down; // Change in option value
            let cu = s_up - s_down; // Change in underlying asset price

            println!("CA (Change in Option Value): {}", ca);
            println!("CU (Change in Underlying Asset Price): {}", cu);

            return ca / cu; // Delta calculation
        }
    }

    // Default return value (should not reach here)
    0.0
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
        let option_kind = OptionKind::Call;
        let option = Option {
            s,
            k,
            t,
            r,
            sigma,
            div_yield,
            steps,
            kind: option_kind,
        };

        // Expected value from a reliable source or previous computation
        let expected_price = 10.969462;
        let calculated_price = binomial_pricing_model(option);

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
        let option_kind = OptionKind::Put;
        let option = Option {
            s,
            k,
            t,
            r,
            sigma,
            div_yield,
            steps,
            kind: option_kind,
        };

        let expected_price = 5.791149;
        let calculated_price = binomial_pricing_model(option);

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
        let option_kind = OptionKind::Call;
        let option = Option {
            s,
            k,
            t,
            r,
            sigma,
            div_yield,
            steps,
            kind: option_kind,
        };

        let expected_price = f64::max(s - k, 0.0); // Intrinsic value
        let calculated_price = binomial_pricing_model(option);

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
        let option_kind = OptionKind::Call;
        let option = Option {
            s,
            k,
            t,
            r,
            sigma,
            div_yield,
            steps,
            kind: option_kind,
        };

        let expected_price = f64::max(s - k, 0.0);
        let calculated_price = binomial_pricing_model(option);

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
        let option_kind = OptionKind::Put;
        let option = Option {
            s,
            k,
            t,
            r,
            sigma,
            div_yield,
            steps,
            kind: option_kind,
        };

        // For deep in-the-money put options, immediate exercise might be optimal
        let expected_price = f64::max(k - s, 0.0);
        let calculated_price = binomial_pricing_model(option);

        assert!(
            (calculated_price - expected_price).abs() < 0.01 || calculated_price > expected_price,
            "Calculated price {} is less than expected intrinsic value {}",
            calculated_price,
            expected_price
        );
    }

    #[test]
    fn test_high_dividend_yield() {
        // Test the effect of a high dividend yield on an call option
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05;
        let sigma = 0.2;
        let div_yield = 0.1; // High dividend yield
        let steps = 100;
        let option_kind = OptionKind::Call;
        let option = Option {
            s,
            k,
            t,
            r,
            sigma,
            div_yield,
            steps,
            kind: option_kind,
        };

        let calculated_price = binomial_pricing_model(option);

        // Without a benchmark, we can at least check that the price is positive
        assert!(
            calculated_price > 0.0,
            "Calculated price should be positive for an call option with high dividend yield"
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
        let option_kind = OptionKind::Call;
        let option = Option {
            s,
            k,
            t,
            r,
            sigma,
            div_yield,
            steps,
            kind: option_kind,
        };

        let calculated_price = binomial_pricing_model(option);

        // The price may not be accurate, but it should be computed without errors
        assert!(
            calculated_price >= 0.0,
            "Calculated price should be non-negative even with low number of steps"
        );
    }

    #[test]
    fn test_american_call_option_delta() {
        // Test parameters
        let s = 100.0; // Current stock price
        let k = 100.0; // Strike price
        let t = 1.0; // Time to expiration in years
        let r = 0.06; // Risk-free interest rate
        let sigma = 0.2; // Volatility
        let div_yield = 0.0; // Dividend yield
        let steps = 100; // Number of steps
        let kind = OptionKind::Call;
        let option = Option {
            s,
            k,
            t,
            r,
            sigma,
            div_yield,
            steps,
            kind,
        };

        let delta = bopm_delta(option);

        // Expected delta from a reliable source or prior computation
        let expected_delta = 0.6716014898;

        let epsilon = 0.01;
        assert!(
            (delta - expected_delta).abs() < epsilon,
            "Calculated delta {} differs from expected delta {}",
            delta,
            expected_delta
        );
    }

    #[test]
    fn test_american_put_option_delta() {
        // Test parameters (aligned with QuantLib)
        let s = 100.0;
        let k = 100.0;
        let t = 1.0;
        let r = 0.05; // Risk-free interest rate adjusted to 0.05
        let sigma = 0.2;
        let div_yield = 0.0;
        let steps = 1000; // Increased number of steps to 1000
        let kind = OptionKind::Put;
        let option = Option {
            s,
            k,
            t,
            r,
            sigma,
            div_yield,
            steps,
            kind,
        };

        let delta = bopm_delta(option);

        // Expected delta from QuantLib
        let expected_delta = -0.4148576844;

        let epsilon = 1e-5;
        assert!(
            (delta - expected_delta).abs() < epsilon,
            "Calculated delta {} differs from expected delta {}",
            delta,
            expected_delta
        );
    }
}
