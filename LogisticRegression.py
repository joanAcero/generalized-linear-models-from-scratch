# This is a custom implementation of Logistic Regression with Statistical Inference
# Authors: Joan Acero, Mateja Zatezalo, Pawarit Jamjod

import numpy as np
from scipy import stats

class CustomLogisticRegression:
    def __init__(self, max_iter=25, tol=1e-6):
        """
        Initialize Logistic Regression model
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of IRLS iterations
        tol : float
            Convergence tolerance for the optimization process
        """
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None
        self.coef_ = None  # Combined coefficients [bias, weights]
        self.losses = []
        self.converged = False
        
        # Statistical inference attributes
        self.vcov_matrix = None      # Variance-covariance matrix
        self.std_errors = None       # Standard errors
        self.z_scores = None         # Z-statistics (Wald test)
        self.p_values = None         # P-values
        self.conf_intervals = None   # Confidence intervals
        self.odds_ratios = None      # Odds ratios
        self.odds_ratios_ci = None   # Odds ratios CI
        
        # Model fit statistics
        self.n_samples = None
        self.n_features = None
        self.log_likelihood_ = None
        self.null_log_likelihood = None
        self.deviance = None
        self.null_deviance = None
        self.aic = None
        self.bic = None
        self.pseudo_r2_mcfadden = None
        
        # Store final weights for inference
        self.final_W = None
        self.X_with_intercept_ = None
    
    def sigmoid(self, z):
        """
        Sigmoid function with numerical stability
            sigmoid(z) = 1 / (1 + e^(-z))
        We include a clipping step to avoid underflow/overflow for very positive/very negative in exp.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_log_likelihood(self, y, p):
        """
        For binary outcomes, each observation follows a Bernoulli distribution:
            P(y_i | p_i) = p_i^y_i * (1 - p_i)^(1-y_i)
        Compute log-likelihood:
            L(β) = ∏[i=1 to n] p_i^y_i * (1 - p_i)^(1-y_i)
            log L(β) = ∑[i=1 to n] [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
        """
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    def fit(self, X, y, feature_names=None):
        """
        Train the logistic regression model using IRLS
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
        feature_names : list of str, optional
            Names of features for interpretation
        """
        # Store dimensions and feature names
        self.n_samples, self.n_features = X.shape
        self.feature_names = feature_names
        
        # Add intercept term (bias) to X
        X_with_intercept = np.column_stack([np.ones(self.n_samples), X])
        self.X_with_intercept_ = X_with_intercept
        
        # Initialize coefficients (bias + weights)
        self.coef_ = np.zeros(self.n_features + 1)
        
        prev_log_likelihood = -np.inf
        
        print("Starting IRLS optimization...")
        
        for iteration in range(self.max_iter):
            # 1. Compute linear predictor: η = Xβ
            eta = np.dot(X_with_intercept, self.coef_)
            
            # 2. Compute predicted probabilities: p = σ(η)
            p = self.sigmoid(eta)
            
            # 3. Compute log-likelihood
            log_likelihood = self.compute_log_likelihood(y, p)
            self.losses.append(-log_likelihood)  # Store negative for loss
            
            # 4. Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged at iteration {iteration}")
                self.converged = True
                break
            
            prev_log_likelihood = log_likelihood
            
            # 5. Compute weights matrix W (diagonal matrix of p(1-p))
            W = p * (1 - p)
            
            # Add small value to prevent singularity
            W = np.maximum(W, 1e-10)
            
            # Store final weights for inference
            self.final_W = W
            
            # 6. Compute working response (adjusted dependent variable)
            # z = η + (y - p) / (p(1-p))
            z = eta + (y - p) / W
            
            # 7. Solve weighted least squares: (X^T W X)β = X^T W z
            # This is the IRLS update step
            
            # Create diagonal weight matrix
            W_matrix = np.diag(W)
            
            # Compute X^T W X (Hessian)
            XtWX = X_with_intercept.T @ W_matrix @ X_with_intercept
            
            # Compute X^T W z
            XtWz = X_with_intercept.T @ (W * z)
            
            # Solve the system (with regularization for numerical stability)
            try:
                # Add small ridge regularization for numerical stability
                ridge = 1e-8 * np.eye(XtWX.shape[0])
                self.coef_ = np.linalg.solve(XtWX + ridge, XtWz)
            except np.linalg.LinAlgError:
                print(f"Singular matrix at iteration {iteration}, stopping.")
                break
            
            print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.6f}")
        
        # Extract bias and weights
        self.bias = self.coef_[0]
        self.weights = self.coef_[1:]
        self.log_likelihood_ = prev_log_likelihood
        
        print(f"\nFinal log-likelihood: {prev_log_likelihood:.6f}")
        print(f"Converged: {self.converged}")
        
        # Compute statistical inference
        self._compute_inference(X_with_intercept, y)
        
        return self
    
    def _compute_inference(self, X_with_intercept, y):
        """
        Compute all statistical inference metrics
        
        Parameters:
        -----------
        X_with_intercept : array, shape (n_samples, n_features + 1)
            Design matrix with intercept column
        y : array, shape (n_samples,)
            Target values
        """
        # 1. Compute Variance-Covariance Matrix
        # Var(β) = (X^T W X)^(-1)
        W_matrix = np.diag(self.final_W)
        hessian = X_with_intercept.T @ W_matrix @ X_with_intercept
        
        try:
            self.vcov_matrix = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            print("Warning: Singular Hessian. Using pseudo-inverse.")
            self.vcov_matrix = np.linalg.pinv(hessian)
        
        # 2. Standard Errors
        # SE(β_j) = sqrt(Var(β_j))
        self.std_errors = np.sqrt(np.diag(self.vcov_matrix))
        
        # 3. Z-scores (Wald statistics)
        # z = β / SE(β)
        # Under H0: β = 0, z ~ N(0,1)
        self.z_scores = self.coef_ / self.std_errors
        
        # 4. P-values (two-tailed)
        # P(|Z| > |z|) where Z ~ N(0,1)
        self.p_values = 2 * (1 - stats.norm.cdf(np.abs(self.z_scores)))
        
        # 5. Confidence Intervals (95% by default)
        z_critical = stats.norm.ppf(0.975)  # 1.96
        margin = z_critical * self.std_errors
        self.conf_intervals = np.column_stack([
            self.coef_ - margin,
            self.coef_ + margin
        ])
        
        # 6. Odds Ratios
        # OR = exp(β)
        self.odds_ratios = np.exp(self.coef_)
        self.odds_ratios_ci = np.exp(self.conf_intervals)
        
        # 7. Model Fit Statistics
        self._compute_fit_statistics(y)
    
    def _compute_fit_statistics(self, y):
        """
        Compute model fit statistics
        
        Parameters:
        -----------
        y : array, shape (n_samples,)
            Target values
        """
        n = self.n_samples
        k = self.n_features + 1  # Number of parameters
        
        # Deviance
        self.deviance = -2 * self.log_likelihood_
        
        # Null model (intercept only)
        p_null = np.mean(y)
        epsilon = 1e-15
        p_null = np.clip(p_null, epsilon, 1 - epsilon)
        self.null_log_likelihood = np.sum(
            y * np.log(p_null) + (1 - y) * np.log(1 - p_null)
        )
        self.null_deviance = -2 * self.null_log_likelihood
        
        # AIC = 2k - 2*log-likelihood
        self.aic = 2 * k - 2 * self.log_likelihood_
        
        # BIC = k*log(n) - 2*log-likelihood
        self.bic = k * np.log(n) - 2 * self.log_likelihood_
        
        # McFadden's Pseudo R²
        self.pseudo_r2_mcfadden = 1 - (self.log_likelihood_ / self.null_log_likelihood)
    
    def summary(self, alpha=0.05):
        """
        Display comprehensive statistical summary
        
        Parameters:
        -----------
        alpha : float
            Significance level (default: 0.05 for 95% CI)
        """
        # Create feature labels
        if self.feature_names:
            labels = ['Intercept'] + list(self.feature_names)
        else:
            labels = ['Intercept'] + [f'X{i}' for i in range(1, self.n_features + 1)]
        
        print("\n" + "="*85)
        print(" "*25 + "LOGISTIC REGRESSION SUMMARY")
        print("="*85)
        
        # Model Information
        print("\nModel Information:")
        print("-"*85)
        print(f"  Number of observations: {self.n_samples:,}")
        print(f"  Number of predictors:   {self.n_features}")
        print(f"  Converged:              {self.converged}")
        print(f"  Iterations:             {len(self.losses)}")
        
        # Goodness-of-Fit
        print("\nGoodness-of-Fit Statistics:")
        print("-"*85)
        print(f"  Log-Likelihood:         {self.log_likelihood_:>12.4f}")
        print(f"  Deviance:               {self.deviance:>12.4f}")
        print(f"  Null Deviance:          {self.null_deviance:>12.4f}")
        print(f"  AIC:                    {self.aic:>12.4f}")
        print(f"  BIC:                    {self.bic:>12.4f}")
        print(f"  McFadden's Pseudo R²:   {self.pseudo_r2_mcfadden:>12.4f}")
        
        # Likelihood Ratio Test
        lr_statistic = -2 * (self.null_log_likelihood - self.log_likelihood_)
        lr_df = self.n_features
        lr_pvalue = 1 - stats.chi2.cdf(lr_statistic, lr_df)
        print(f"\n  Likelihood Ratio Test:  χ²({lr_df}) = {lr_statistic:.4f}, p = {lr_pvalue:.4e}")
        
        # Coefficients Table
        ci_level = int((1 - alpha) * 100)
        print(f"\nCoefficients (with {ci_level}% Confidence Intervals):")
        print("="*85)
        print(f"{'Variable':<15} {'Coef':>10} {'Std.Err':>10} {'z':>8} "
              f"{'P>|z|':>10} {'[{:.3f}'.format(alpha/2):>10} {'{:.3f}]'.format(1-alpha/2):>10} {'':>3}")
        print("-"*85)
        
        for i, label in enumerate(labels):
            stars = self._get_significance_stars(self.p_values[i])
            print(f"{label:<15} {self.coef_[i]:>10.4f} {self.std_errors[i]:>10.4f} "
                  f"{self.z_scores[i]:>8.3f} {self.p_values[i]:>10.4f} "
                  f"{self.conf_intervals[i,0]:>10.4f} {self.conf_intervals[i,1]:>10.4f} {stars:>3}")
        
        print("-"*85)
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        
        # Odds Ratios
        print(f"\nOdds Ratios (with {ci_level}% Confidence Intervals):")
        print("="*85)
        print(f"{'Variable':<15} {'OR':>10} {'[{:.3f}'.format(alpha/2):>10} {'{:.3f}]'.format(1-alpha/2):>10} {'Interpretation':<35}")
        print("-"*85)
        
        for i, label in enumerate(labels):
            interpretation = self._interpret_odds_ratio(self.odds_ratios[i], label)
            print(f"{label:<15} {self.odds_ratios[i]:>10.4f} "
                  f"{self.odds_ratios_ci[i,0]:>10.4f} {self.odds_ratios_ci[i,1]:>10.4f} "
                  f"{interpretation:<35}")
        
        print("="*85)
        
        # Interpretation Guide
        print("\nInterpretation Guide:")
        print("-"*85)
        print("• Coefficients (β): Change in log-odds for 1-unit increase in predictor")
        print("• Odds Ratios (OR): Multiplicative change in odds for 1-unit increase")
        print("    - OR > 1: Increases odds of outcome")
        print("    - OR < 1: Decreases odds of outcome")
        print("    - OR = 1: No effect")
        print("• Z-statistic: Tests H₀: β = 0")
        print("• P-value: Probability of observing coefficient under H₀")
        print("• Pseudo R²: 0.2-0.4 indicates excellent fit (McFadden)")
        print("="*85 + "\n")
    
    def _get_significance_stars(self, p):
        """Get significance stars for p-value"""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "."
        return ""
    
    def _interpret_odds_ratio(self, or_value, var_name):
        """Generate interpretation for odds ratio"""
        if or_value > 1.5:
            return f"Strong positive effect"
        elif or_value > 1.1:
            return f"Moderate positive effect"
        elif or_value > 0.95:
            return f"Weak/no effect"
        elif or_value > 0.67:
            return f"Moderate negative effect"
        else:
            return f"Strong negative effect"
    
    def get_inference_dict(self):
        """
        Return inference results as dictionary
        
        Returns:
        --------
        dict : Dictionary with all inference statistics
        """
        if self.feature_names:
            labels = ['Intercept'] + list(self.feature_names)
        else:
            labels = ['Intercept'] + [f'X{i}' for i in range(1, self.n_features + 1)]
        
        return {
            'coefficients': dict(zip(labels, self.coef_)),
            'std_errors': dict(zip(labels, self.std_errors)),
            'z_scores': dict(zip(labels, self.z_scores)),
            'p_values': dict(zip(labels, self.p_values)),
            'conf_intervals_95': dict(zip(labels, self.conf_intervals.tolist())),
            'odds_ratios': dict(zip(labels, self.odds_ratios)),
            'odds_ratios_ci_95': dict(zip(labels, self.odds_ratios_ci.tolist())),
            'vcov_matrix': self.vcov_matrix,
            'model_fit': {
                'log_likelihood': self.log_likelihood_,
                'deviance': self.deviance,
                'null_deviance': self.null_deviance,
                'aic': self.aic,
                'bic': self.bic,
                'pseudo_r2_mcfadden': self.pseudo_r2_mcfadden
            }
        }
    
    def predict_proba(self, X):
        """
        Predict probability estimates
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
            
        Returns:
        --------
        probabilities : array, shape (n_samples,)
            Predicted probabilities
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test data
        threshold : float
            Classification threshold
            
        Returns:
        --------
        predictions : array, shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_coefficients(self):
        """
        Get model coefficients
        """
        return {
            'bias': self.bias,
            'weights': self.weights,
            'all_coefficients': self.coef_
        }


# Example usage
if __name__ == "__main__":
    # Generate sample data with meaningful features
    np.random.seed(42)
    
    # Create features: Age and Income (scaled)
    n_samples = 100
    age = np.random.normal(40, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    
    # Normalize features
    X = np.column_stack([
        (age - age.mean()) / age.std(),
        (income - income.mean()) / income.std()
    ])
    
    # Generate target: probability increases with age and income
    z = -2 + 1.5 * X[:, 0] + 0.8 * X[:, 1]
    prob = 1 / (1 + np.exp(-z))
    y = (np.random.random(n_samples) < prob).astype(int)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    feature_names = ['Age', 'Income']
    # Train model
    model = CustomLogisticRegression(max_iter=100, tol=1e-6)
    model.fit(X_train, y_train, feature_names=feature_names)
    model.summary()

    # Predictions
    y_pred = model.predict(X_test)
    print(f"Test Predictions: {y_pred}")
    print(f"Test Actuals:     {y_test}")

    # Plot original data
    import matplotlib.pyplot as plt
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
    plt.xlabel('Age (normalized)')
    plt.ylabel('Income (normalized)')
    plt.title('Logistic Regression Data Distribution')
    plt.savefig('logistic_regression_data.png')

    # Plot test data labels
    plt.figure()
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, alpha=0.7)
    plt.xlabel('Age (normalized)')
    plt.ylabel('Income (normalized)')
    plt.title('Test Data Actual Labels')
    plt.savefig('logistic_regression_test_actuals.png')

