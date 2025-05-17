import pandas as pd
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import namedtuple
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF # WhiteKernel, ConstantKernel

train_colour = "#1f77b4"
test_colour = "#ff7f0e"

def quadratic_mean(series):
    return np.sqrt(np.mean(series**2))

def group_by_mean_squared(df: pd.DataFrame, group_by: str | list[str]) -> pd.DataFrame:
    grouped_df = df.groupby(by=group_by).agg(quadratic_mean).reset_index()
    return grouped_df

def transform_direct(y, transform_type, transform_params = None):
    if transform_type == "logp":
        return np.log1p(y), None
    elif transform_type == "log":
        return np.log(y), None
    elif transform_type == "box-cox":
        if transform_params is None:
            y, lambda_box_cox = scipy.stats.boxcox(y)
            return y, lambda_box_cox
        else:
            y = scipy.stats.boxcox(y, lmbda=transform_params)
            return y, transform_params
    elif transform_type is None:
        return y, None
    else:
        raise ValueError("Invalid transform type. Choose 'logp' or 'log'.")

def transform_inverse(y, transform_type, transform_params = None):
    if transform_type == "logp":
        return np.expm1(y)
    elif transform_type == "log":
        return np.exp(y)
    elif transform_type == "box-cox":
        return scipy.special.inv_boxcox(y, transform_params)
    elif transform_type is None:
        return y
    else:
        raise ValueError("Invalid transform type. Choose 'logp' or 'log'.")

def predict_with_model(this_model: dict, X: pd.DataFrame) -> pd.DataFrame:
    y_transformed_scaled = this_model["model"].predict(X).reshape(-1, 1)
    y_transformed = this_model["y_scaler"].inverse_transform(y_transformed_scaled).flatten()
    return transform_inverse(y_transformed, this_model["transform_type"], this_model["transform_params"])

def remove_ouliers(df: pd.DataFrame, target_name: str, outlier_detection: str) -> tuple[pd.DataFrame, pd.Series]:
    if outlier_detection in ["IQR", "IQR+"]:
        # Remove outliers based on the target variable
        # Calculate the IQR
        y_raw = df[target_name]
        Q1 = np.percentile(y_raw, 25)  # First quartile (25th percentile)
        Q3 = np.percentile(y_raw, 75)  # Third quartile (75th percentile)
        IQR = Q3 - Q1  # Interquartile range

        if outlier_detection == "IQR":
            # Define the lower and upper bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outlier_mask = (y_raw < lower_bound) | (y_raw > upper_bound)
            outliers = y_raw[outlier_mask]
            print(f"Removing {len(outliers)} outliers using {outlier_detection} method with {target_name} < {lower_bound:.4f} and {target_name} > {upper_bound:.4f}")
        else:
            # Define the upper bound for outliers
            upper_bound = Q3 + 1.5 * IQR
            # Identify outliers
            outlier_mask = (y_raw > upper_bound)
            outliers = y_raw[outlier_mask]
            print(f"Removing {len(outliers)} outliers using {outlier_detection} method with {target_name} > {upper_bound:.4f}")

        df_no_outliers = df[~outlier_mask].reset_index(drop=True)
        
    elif outlier_detection is None:
        df_no_outliers = df.copy()
        outliers = pd.Series(dtype=float)
        print(f"No outliers removed using {outlier_detection} method")
    else:
        raise ValueError("Invalid outlier detection method. Choose 'IQR', 'IQR+', or None.")
    return df_no_outliers, outliers

def set_axis_grid_style(ax: plt.axis) -> plt.axis:
    ax.grid(which="major", alpha=0.35)
    ax.grid(which="minor", alpha=0.3, linestyle=':')
    ax.minorticks_on()
    return ax

ParameterRange = namedtuple("ParameterRange", ["start", "end", "name"])
ParameterSlice = namedtuple("ParameterRange", ["range", "name"])

def sweep_2d(x1: ParameterRange, x2: ParameterRange, x_fixed: dict[str, float], model_result: dict, label_mapping: dict[str, str], x_slice: ParameterSlice | None = None) -> None:
    target_name = model_result["target_name"]

    # Predict over a grid
    n = 31
    y_0_range = np.linspace(0, 0.8, n)
    tau_range = np.linspace(0.5, 2, n)
    a_grid, b_grid = np.meshgrid(y_0_range, tau_range)
    X_grid_raw = np.column_stack((a_grid.ravel(), b_grid.ravel()))
    X_grid_raw = pd.DataFrame(X_grid_raw, columns=[x1.name, x2.name])

    if x_slice is None:
        x_slice_name = list(x_fixed.keys())[0]
        x_slice = ParameterSlice(range=[x_fixed[x_slice_name]], name=x_slice_name)
        x_fixed = {k: v for k, v in x_fixed.items() if k != x_slice.name}

    X_grids = list()
    for slice_value in x_slice.range:
        X_grid_slice = X_grid_raw.copy()
        X_grid_slice[x_slice.name] = slice_value
        for fixed_param, fixed_value in x_fixed.items():
            X_grid_slice[fixed_param] = fixed_value
        X_grids.append(X_grid_slice)

    fig = plt.figure(figsize=(12, 8), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    for X_grid in X_grids:
        for fixed_param, fixed_value in x_fixed.items():
            X_grid[fixed_param] = fixed_value

        # Reorder the columns to match the model's input
        cols = X_grid.columns.tolist()
        ordered = model_result["features"]
        new = [item for item in ordered if item in cols]
        X_grid = X_grid[new]

        X_grid_scaled = model_result["X_scaler"].transform(X_grid)
        if "poly" in model_result:
            X_ready = model_result["poly"].transform(X_grid_scaled)
        else:
            X_ready = X_grid_scaled

        y_pred = predict_with_model(model_result, X_ready).reshape(n, n)

        # Plot
        ax.plot_surface(a_grid, b_grid, y_pred, cmap="coolwarm")

    ax.set_xlabel(f"{label_mapping[x1.name]}")
    ax.set_ylabel(f"{label_mapping[x2.name]}")
    ax.set_zlabel(f"{label_mapping[target_name]}")
    ax.set_zlim(0)
    plt.title(f"Error surface for {label_mapping[target_name]}")
    plt.show()

def stack_images_vertically(images: list[str], save_file_path: str) -> None:
    # Load all of the images
    images = [Image.open(image_path) for image_path in images]

    # Assume all images are the same size
    width, height = images[0].size

    # Create a new blank image with size to hold a 2x2 grid
    combined = Image.new("RGBA", (width, 4 * height))

    # Paste the 4 images into the correct positions
    for imgage_idx, image in enumerate(images):
        combined.paste(image, (0, imgage_idx*height))

    # Save the combined image
    combined.save(save_file_path)

def plot_histogram(df_agg: pd.DataFrame, params_fitted: list[str], label_mapping: dict[str, str], save_filepath: str | None = None, dpi : float = 150) -> None:
    n_params = len(params_fitted)
    fig, ax = plt.subplots(n_params, 3, figsize=(14, 14), tight_layout=True)
    for idx, param in enumerate(params_fitted):
        param_error = param + "_error"
        y = df_agg[param_error]
        ax[idx][0].hist(y, bins="auto", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax[idx][0].axvline(y.mean(), color="red", linestyle="--", label="Mean")
        ax[idx][0].axvline(y.median(), color="blue", linestyle="--", label="Median")
        ax[idx][0].set_xlabel(label_mapping[param_error])

        y_log = np.log(y)
        ax[idx][1].hist(y_log, bins="auto", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax[idx][1].axvline(y_log.mean(), color="red", linestyle="--", label="Mean")
        ax[idx][1].axvline(y_log.median(), color="blue", linestyle="--", label="Median")
        ax[idx][1].set_xlabel("log(" + label_mapping[param_error] + ")")
        
        y_box_cox, lambda_param = scipy.stats.boxcox(y)
        ax[idx][2].hist(y_box_cox, bins="auto", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax[idx][2].axvline(y_box_cox.mean(), color="red", linestyle="--", label="Mean")
        ax[idx][2].axvline(np.median(y_box_cox), color="blue", linestyle="--", label="Median")
        ax[idx][2].set_xlabel(fr"boxcox({label_mapping[param_error]}) with $\lambda$ = {lambda_param:.3f}")

        for a in ax[idx]:
            a.set_ylabel("Frequency")
            a.legend()
            a = set_axis_grid_style(a)

    fig.suptitle("""Histogram of the fitted parameters' errors, their "log" and "Box-Cox" transformations""", fontsize=16)

    if save_filepath:
        plt.savefig(save_filepath, dpi=dpi, bbox_inches="tight")
    plt.show()


label_mapping = {
    "y_0": r"$y_0$",
    "B_z_0": r"$B_z^{0}$",
    "tau": r"$\tau$",
    "C_nm": r"$C_{nm}$",
    "delta": r"$\delta$",
    "psi": r"$\psi$",
    "y_0_error": r"$y_0^{error}$",
    "B_z_0_error": r"$B_z^{0, error}$",
    "tau_error": r"$\tau^{error}$",
    "C_nm_error": r"$C_{nm}^{error}$",
    "delta_error": r"$\delta^{error}$",
    "psi_error": r"$\psi^{error}$",
    "y_0_rel_error": r"$y_0^{rel\ error}$",
    "B_z_0_rel_error": r"$B_z^{0, rel\ error}$",
    "tau_rel_error": r"$\tau^{rel\ error}$",
    "C_nm_rel_error": r"$C_{nm}^{rel\ error}$",
    "delta_rel_error": r"$\delta^{rel\ error}$",
    "psi_rel_error": r"$\psi^{rel\ error}$",
    "y_0_opt": r"$y_0^{opt}$",
    "B_z_0_opt": r"$B_z^{0,\ opt}$",
    "tau_opt": r"$\tau^{opt}$",
    "C_nm_opt": r"$C_{nm}^{opt}$",
    "delta_opt": r"$\delta^{opt}$",
    "psi_opt": r"$\psi^{opt}$",
    "noise_level": r"$\sigma$"
}

def plot_target_error_and_relative_correlations(df_agg: pd.DataFrame, params_fitted: list, label_mapping: dict, corr_method: str = "pearson", save_file_name: str | None = None) -> None:
    df_corr = df_agg[[p + "_error" for p in params_fitted]].corr(method=corr_method)
    n = df_corr.shape[0]
    lower_triangular_mask = np.tril(np.ones((n, n)), k=-1).astype(bool)
    df_corr = df_corr.where(lower_triangular_mask)
    df_corr = pd.DataFrame(df_corr.values[1 : n, 0 : n-1], index=df_corr.index[1 : n], columns=df_corr.columns[0 : n-1])

    # Show the correlation matrix of the relative errors
    df_rel_corr = df_agg[[p + "_rel_error" for p in params_fitted]].corr(method=corr_method)
    df_rel_corr = df_rel_corr.where(lower_triangular_mask)
    df_rel_corr = pd.DataFrame(df_rel_corr.values[1 : n, 0 : n-1], index=df_rel_corr.index[1 : n], columns=df_rel_corr.columns[0 : n-1])

    # Rename both index and columns
    df_corr = df_corr.rename(index=label_mapping, columns=label_mapping)
    df_rel_corr = df_rel_corr.rename(index=label_mapping, columns=label_mapping)

    v_min = min(df_corr.min().min(), df_rel_corr.min().min())
    v_max = min(df_corr.max().max(), df_rel_corr.max().max())

    fig, axis = plt.subplots(1, 2, figsize=(14, 7))
    labels = ["", " relative"]
    for ax, df, this_label in zip(axis, [df_corr, df_rel_corr], labels):
        sns.heatmap(df, annot=True, fmt=".3f", cmap="coolwarm", vmin=v_min, vmax=v_max, cbar_kws={"shrink": 1.5}, square=True, linewidths=0.5, cbar=False, ax=ax)
        ax.set_title(f"Correlation ({corr_method.capitalize()}) matrix of the" + this_label + " fitted parameters' errors")
    
    if save_file_name:
        fig.savefig(save_file_name, dpi=200, bbox_inches="tight")
    plt.show()

def plot_target_error_and_relative_vs_features_correlations(df_agg: pd.DataFrame, params_fitted: list, label_mapping: dict, corr_method: str = "pearson", save_file_name: str | None = None) -> None:
    fig, axis = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    target_explnatory_correlation_matrix = {}

    for ax, error_type in zip(axis, ["_error", "_rel_error"]):
        df_target_corr = []
        for param in params_fitted:
            explainatory_variables = [param + error_type] + [p + "_opt" for p in params_fitted] + ["noise_level"]
            df_corr = df_agg[explainatory_variables].corr(method=corr_method)
            df_corr = pd.DataFrame(df_corr.values[1 :, 0], index=df_corr.index[1 :], columns=df_corr.columns[0 : 1])
            df_target_corr.append(df_corr)

        df_target_corr = pd.concat(df_target_corr, axis=1)
        df_target_corr = df_target_corr.rename(index=label_mapping, columns=label_mapping)
        target_explnatory_correlation_matrix[error_type] = df_target_corr

    v_min = min(df_target_corr.min().min() for df_target_corr in target_explnatory_correlation_matrix.values())
    v_max = max(df_target_corr.max().max() for df_target_corr in target_explnatory_correlation_matrix.values())

    for ax, df_target_corr in zip(axis, target_explnatory_correlation_matrix.values()):
        sns.heatmap(df_target_corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=v_min, vmax=v_max, cbar_kws={"shrink": 1.5}, square=True, linewidths=0.5, cbar=False, ax=ax)

    plt.suptitle(f"Correlation ({corr_method.capitalize()}) matrices of the error and relative error of the fitted parameters v.s. explanatory variables")
    
    if save_file_name is not None:
        fig.savefig(save_file_name, dpi=200, bbox_inches="tight")

    plt.show()

def fit_regression_model(df: pd.DataFrame, params_fitted: list[str], target_name: str, model_type: str, transform_type: str | None, outlier_detection: str | None, verbose: bool = False, random_state: int=42):
    # Define the features and target
    x_features = [p + "_opt" for p in params_fitted] + ["noise_level"]

    model_result = {"features": x_features,
                    "target_name": target_name,
                    "transform_type": transform_type,
                    "outlier_detection": outlier_detection,
                    "model_type": model_type,}

    df_no_outliers, outliers = remove_ouliers(df, target_name, outlier_detection)
    model_result["outliers"] = outliers

    # Split the explicative and target variables
    X = df_no_outliers[x_features]
    y = df_no_outliers[target_name].to_numpy()

    # Scale the explicative variables
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)
    model_result["X_scaler"] = X_scaler
    model_result["X_scaled"] = X_scaled

    # Create polynomial features of degree 2 with cross-terms
    if model_type in ["linear", "ridge"]:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_ready = poly.fit_transform(X_scaled)
    else:
        X_ready = X_scaled

    # Transform the target variable
    y_transformed, transform_params = transform_direct(y, transform_type)
    model_result["transform_params"] = transform_params

    # Scale the target variable (this needs to be done after the transformation)
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_transformed.reshape(-1, 1)).flatten()
    model_result["y_scaler"] = y_scaler

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_ready, y_scaled, test_size=0.3, random_state=random_state)

    if model_type == "linear":
        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif model_type == "ridge":
        # Train a Ridge regression model with alpha grid search
        param_grid = {'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 2, 5, 10]}
        ridge_cv = GridSearchCV(Ridge(), param_grid, cv=5, scoring="r2")
        ridge_cv.fit(X_train, y_train)
        best_alpha = ridge_cv.best_params_['alpha']

        print(f"Best alpha: {best_alpha}")
        print(f"Best cross-validated R^2: {ridge_cv.best_score_}")

        model = ridge_cv.best_estimator_
    elif model_type == "gpr":

        param_grid = {
            "alpha": [1e-10, 1e-5, 1e-2, 1e-1, 0.5, 1, 2],
            "kernel": [RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 50)) for length_scale in [0.1, 0.5, 1, 5, 10]]
        }

        # Define the GPR model
        gpr = GaussianProcessRegressor(n_restarts_optimizer=5, normalize_y=True) # kernel=RBF(length_scale=50), alpha=1e-2, 
        
        grid_search = GridSearchCV(estimator=gpr, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validated RMSE: {math.sqrt(-grid_search.best_score_)}")

        model.fit(X_train, y_train)
    else:
        raise ValueError("Invalid model type. Choose 'linear' or 'ridge'.")

    model_result["model"] = model

    # Evaluate the model (training set)
    y_pred_train = predict_with_model(model_result, X_train)
    y_train_inv_transformed = transform_inverse(y_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten(), transform_type, transform_params)
    if not np.any(np.isnan(y_pred_train)):
        rmse_train = math.sqrt(mean_squared_error(y_train_inv_transformed, y_pred_train))
        r2_train = r2_score(y_train_inv_transformed, y_pred_train)
    else:
        rmse_train = np.nan
        r2_train = np.nan

    # Evaluate the model (test set)
    y_pred_test = predict_with_model(model_result, X_test)
    y_test_inv_transformed = transform_inverse(y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(), transform_type, transform_params)
    if not np.any(np.isnan(y_pred_test)):
        rmse_test = math.sqrt(mean_squared_error(y_test_inv_transformed, y_pred_test))
        r2_test = r2_score(y_test_inv_transformed, y_pred_test)
    else:
        rmse_test = np.nan
        r2_test = np.nan

    # Calculate the residuals on the transformed space, that is, where the model was trained
    model_result["train_residuals"] = model.predict(X_train) - y_train
    model_result["test_residuals"] = model.predict(X_test) - y_test

    stats = pd.DataFrame({
        "RMSE": [rmse_train, rmse_test],
        "R^2": [r2_train, r2_test]
    }, index=["Train", "Test"])

    model_result.update({           
        "transform_params": transform_params,
        "outlier_detection": outlier_detection,
        "outliers": outliers,
        "features": x_features,
        "target_name": target_name,
        "X_train": X_train,
        "y_train": y_train,
        "y_train_raw": y_train_inv_transformed,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_raw": y_test_inv_transformed,
        "stats": stats})

    if model_type == "ridge":
        model_result["best_alpha"] = best_alpha
    
    # Compute a model label
    if model_type == "ridge":
        model_label = rf"{model_type} ($\alpha$={best_alpha})"
    elif model_type == "gpr":
        model_label = rf"{model_type} ($\alpha$={model.alpha:.3f}, $l$={model.kernel_.length_scale:.3f})"
    else:
        model_label = model_type

    if transform_type == "box-cox":
        transform_label = rf"{transform_type} ($\lambda$={transform_params:.3f})"
    else:
        transform_label = transform_type
    model_result["model_label"] = f"{model_label}-{transform_label}-{outlier_detection}"

    # Display the coefficients and feature names
    if verbose:
        print(f"Train RSME: {rmse_train}")
        print(f"Train R^2 Score: {r2_train}")

        print(f"Test RSME: {rmse_test}")
        print(f"Test R^2 Score: {r2_test}")

        feature_names = poly.get_feature_names_out(x_features)
        coefficients = model.coef_
        intercept = model.intercept_

        print(f"Model coefficients ({len(coefficients) + 1}):")
        for name, coef in zip(feature_names, coefficients):
            print(f"  -> {name}: {coef}")
        print(f"  -> Intercept: {intercept}")   
    
    if model_type in ["linear", "ridge"]:
        model_result["poly"] = poly

    return model_result

def plot_model_results(models: dict, save_file_name: str | None = None) -> None:
    n_cols = 2
    n_rows = math.ceil(len(models) / n_cols)
    fig, axis = plt.subplots(n_rows, n_cols, figsize=(12, 12), tight_layout=True)

    for plot_idx, (param, model_result) in enumerate(models.items()):
        ax = axis[plot_idx // n_cols][plot_idx % n_cols]
        ax.scatter(model_result["y_train_raw"], predict_with_model(model_result, model_result["X_train"]), alpha=0.3, label="Train")
        ax.scatter(model_result["y_test_raw"], predict_with_model(model_result, model_result["X_test"]), alpha=0.3, label="Test")
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against eachother
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
        ax.grid(alpha=0.35)
        ax.set_xlabel(f"Actual {label_mapping[param]}")
        ax.set_ylabel(f"Predicted {label_mapping[param]}")
        ax.set_title(f"{label_mapping[param]}: {model_result["model_label"]}")
        ax.legend()

    if save_file_name:
        plt.savefig(save_file_name, dpi=200, bbox_inches="tight")
    plt.show()

def get_models_stats(models: dict) -> pd.DataFrame:
    stats_general = {}
    for param_error_name in models:
        df_stats = models[param_error_name]["stats"]
        stats_general[param_error_name] = [df_stats["RMSE"]["Train"],
                                            df_stats["RMSE"]["Test"],
                                            df_stats["R^2"]["Train"],   
                                            df_stats["R^2"]["Test"]]

    return pd.DataFrame(stats_general, index=["RMSE Train", "RMSE Test", "R^2 Train", "R^2 Test"])

def plot_residuals(models: dict, method: str, save_file_path: str | None = None) -> None:
    n_cols = 2
    n_rows = math.ceil(len(models) / n_cols)
    fig, axis = plt.subplots(n_rows, n_cols, figsize=(12, 12), tight_layout=True)

    for plot_idx, (param, model_result) in enumerate(models.items()):
        ax = axis[plot_idx // n_cols][plot_idx % n_cols]
        if method == "histogram":
            train_mean = model_result["train_residuals"].mean()
            test_mean = model_result["test_residuals"].mean()
            ax.hist(model_result["train_residuals"], bins="auto", facecolor=train_colour, label="Train", alpha=0.6, edgecolor="black", linewidth=0.5)
            ax.hist(model_result["test_residuals"], bins="auto", facecolor=test_colour, label="Test", alpha=0.6, edgecolor="black", linewidth=0.5)
            ax.axvline(train_mean, color=train_colour, linestyle="--", label=f"Train mean = {train_mean:.3f}")
            ax.axvline(test_mean, color=test_colour, linestyle="--", label=f"Test mean = {test_mean:.3f}")
            ax.set_xlabel(f"{label_mapping[param]} residual")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram of residuals of {label_mapping[param]} fitted with {model_result["model_label"]}", fontsize=10)
            ax.legend()
        elif method == "Q-Q":
            scipy.stats.probplot(model_result["train_residuals"], dist="norm", plot=ax)
            # stats.probplot(model_result["test_residuals"], dist="norm", plot=ax)
        else:
            raise ValueError("Invalid method. Choose 'histogram' or 'Q-Q'.")

        ax = set_axis_grid_style(ax)
    
    if save_file_path:
        plt.savefig(save_file_path, dpi=150, bbox_inches="tight")
    plt.show()