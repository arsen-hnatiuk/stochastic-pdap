import numpy as np
import scipy as sp
import pandas as pd
import logging
from sissopp import FeatureSpace
from sissopp.py_interface import import_dataframe
from lib.default_values import *
from lib.ssn import SSN

logging.basicConfig(
    level=logging.DEBUG,
)


class SPDAP_Finite:
    # An implementation of the stochastic sampling approach to finite LGCG

    def __init__(
        self,
        alpha: float,
        data_path: str,
        test_power: float = 0.05,
        parameter_increase: float = 0.1,
    ) -> None:
        target = pd.read_csv(data_path)["log kappa_L"].to_numpy()
        self.target_norm = np.linalg.norm(target)
        self.target = target / self.target_norm
        self.alpha = alpha
        # self.test_power = test_power
        # self.parameter_increase = parameter_increase
        self.feature_inputs = import_dataframe.create_inputs(
            df="thermal_conductivity_data.csv",
            max_rung=3,
            max_param_depth=0,
            prop_key="log kappa_L",
            calc_type="regression",
            n_sis_select=10,
            allowed_ops=[],
            n_rung_generate=0,
            n_rung_store=-1,
            allowed_param_ops=[
                "add",
                "sub",
                "abs_diff",
                "mult",
                "div",
                "inv",
                "abs",
                "exp",
                "log",
                "sin",
                "cos",
                "sq",
                "cb",
                "six_pow",
                "sqrt",
                "cbrt",
                "neg_exp",
            ],
            global_param_opt=False,
            reparam_residual=False,
        )
        self.g = get_default_g(self.alpha)
        self.L = 1
        self.j = lambda K, u: 0.5 * np.linalg.norm(
            np.matmul(K, u) - self.target
        ) ** 2 + self.g(u)
        self.rho = lambda K, u: np.matmul(K, u) - self.target
        self.M = (
            self.j(np.zeros((len(self.target), 1)), np.zeros(1)) / self.alpha
        )  # Bound on the norm of iterates
        self.C = 4 * self.L * self.M**2
        self.machine_precision = 1e-12

    # def explicit_Phi(
    #     self,
    #     rho: np.ndarray,
    #     u: np.ndarray,
    #     v: np.ndarray,
    #     observation_u: np.ndarray,
    #     observation_v: np.ndarray,
    # ) -> float:
    #     # -<rho,K(v-u)>+g(u)-g(v)
    #     return -np.dot(rho, observation_v - observation_u) + self.g(u) - self.g(v)

    # def Phi(
    #     self,
    #     rho: np.ndarray,
    #     u: np.ndarray,
    #     observation_u: np.ndarray,
    #     observation_v: np.ndarray,
    # ) -> float:
    #     # M*max{0,||p_u(x)||-alpha}+g(u)-<rho,Ku>
    #     return (
    #         self.M * (max(0, np.abs(np.dot(rho, observation_v)) - self.alpha))
    #         + self.g(u)
    #         - np.dot(rho, observation_u)
    #     )

    # def rejection_probability(self, beta: float, beta_plus: float) -> float:
    #     sin_square = np.sin(np.arccos(beta)) ** 2
    #     sin_square_plus = np.sin(np.arccos(beta_plus)) ** 2
    #     a = 0.5 * (len(self.target) - 1)
    #     return (
    #         sp.special.betainc(a, 0.5, sin_square)
    #         - sp.special.betainc(a, 0.5, sin_square_plus)
    #     ) / (1 - sp.special.betainc(a, 0.5, sin_square_plus))

    # def get_sample_size(self, rejection_probability: float) -> int:
    #     size = 1
    #     iterate = 1 - rejection_probability
    #     while iterate > self.test_power:
    #         size += 1
    #         iterate *= 1 - rejection_probability
    #     return size

    # def choose_x(
    #     self, rho: np.ndarray, observation: np.ndarray, u: np.ndarray, epsilon: float
    # ) -> tuple:
    #     feature_space = FeatureSpace(self.feature_inputs)
    #     sample_features_raw = np.array([feature.value for feature in feature_space.phi])
    #     sample_norms = np.linalg.norm(sample_features_raw, axis=1)
    #     sample_features = np.array(
    #         [feature / norm for feature, norm in zip(sample_features_raw, sample_norms)]
    #     )
    #     total_sample = len(sample_features)
    #     sample_values = np.abs(np.matmul(sample_features, rho))
    #     incumbent = np.argmax(sample_values)
    #     incumbent_dict = {
    #         "feature": sample_features[incumbent],
    #         "feature_norm": sample_norms[incumbent],
    #         "expression": feature_space.phi[incumbent].expr,
    #     }
    #     Phi_value = self.explicit_Phi(
    #         rho=rho,
    #         u=u,
    #         v=np.array([self.M]),
    #         observation_u=observation,
    #         observation_v=self.M
    #         * np.sign(np.dot(rho, incumbent_dict["feature"]))
    #         * incumbent_dict["feature"],
    #     )
    #     incumbent_dict["Phi_value"] = Phi_value
    #     if Phi_value >= self.M * epsilon:
    #         return incumbent_dict, True
    #     beta = sample_values[incumbent] / np.linalg.norm(rho)
    #     beta_plus = min(beta / (1 - self.parameter_increase), 1)
    #     accept = False
    #     while beta_plus < 1 and accept == False:
    #         probability = self.rejection_probability(beta, beta_plus)
    #         sample_size = self.get_sample_size(rejection_probability=probability)
    #         while total_sample < sample_size:
    #             feature_space = FeatureSpace(self.feature_inputs)
    #             sample_features_raw = np.array(
    #                 [feature.eval for feature in feature_space.phi]
    #             )
    #             sample_norms = np.linalg.norm(sample_features_raw, axis=1)
    #             sample_features = np.array(
    #                 [
    #                     feature / norm
    #                     for feature, norm in zip(sample_features_raw, sample_norms)
    #                 ]
    #             )
    #             total_sample += len(sample_features)
    #             sample_values = np.abs(np.matmul(sample_features, rho))
    #             possible_incumbent = np.argmax(sample_values)
    #             if sample_values[possible_incumbent] / np.linalg.norm(rho) > beta:
    #                 incumbent = possible_incumbent
    #                 incumbent_dict = {
    #                     "feature": sample_features[incumbent],
    #                     "feature_norm": sample_norms[incumbent],
    #                     "expression": feature_space.phi[incumbent].expr,
    #                 }
    #                 Phi_value = self.explicit_Phi(
    #                     rho=rho,
    #                     u=u,
    #                     v=np.array([self.M]),
    #                     observation_u=observation,
    #                     observation_v=self.M
    #                     * np.sign(np.dot(rho, incumbent_dict["feature"]))
    #                     * incumbent_dict["feature"],
    #                 )
    #                 incumbent_dict["Phi_value"] = Phi_value
    #                 if Phi_value >= self.M * epsilon:
    #                     return incumbent_dict, True
    #                 beta = sample_values[incumbent] / np.linalg.norm(rho)
    #                 beta_plus = min(beta / (1 - self.parameter_increase), 1)
    #                 continue
    #         accept = True
    #     return incumbent_dict, False

    def compute_probability(self, beta: float, cutoff: float) -> float:
        # Compute the probability of P_beta[w<=cutoff]
        sin_square = np.sin(np.arccos(beta)) ** 2
        sin_square_cutoff = np.sin(np.arccos(cutoff)) ** 2
        a = 0.5 * (len(self.target) - 2)
        return min(
            (1 - sp.special.betainc(a, 0.5, sin_square_cutoff))
            / (1 - sp.special.betainc(a, 0.5, sin_square)),
            1,
        )

    def compute_sample_size(self, probability: float, tolerance: float) -> int:
        size = 1
        iterate = probability
        while iterate > tolerance:
            size += 1
            iterate *= probability
        return size

    def find_x_tilde(
        self, rho: np.ndarray, tol: float, probability_tolerance: float
    ) -> tuple:
        incumbent_dict = {"value": -1}
        generated_sample_size = 0
        null_beta = self.apha * (1 + tol) / np.linalg.norm(rho)
        if null_beta >= 1:
            # The tolerance has been reached
            return incumbent_dict, generated_sample_size
        alternative_beta = self.apha / np.linalg.norm(rho)
        interval_probability = self.compute_probability(null_beta, alternative_beta)
        sample_size = self.compute_sample_size(
            interval_probability, probability_tolerance
        )
        while (
            generated_sample_size < sample_size
            and incumbent_dict["value"] <= alternative_beta
        ):
            feature_space = FeatureSpace(self.feature_inputs)
            sample_features_raw = np.array(
                [feature.value for feature in feature_space.phi]
            )
            sample_norms = np.linalg.norm(sample_features_raw, axis=1)
            sample_features = np.array(
                [
                    feature / norm
                    for feature, norm in zip(sample_features_raw, sample_norms)
                ]
            )
            generated_sample_size += len(sample_features)
            sample_values = np.abs(np.matmul(sample_features, rho))
            incumbent = np.argmax(sample_values)
            max_value = sample_values[incumbent]
            if max_value > alternative_beta:
                incumbent_dict["value"] = max_value
                incumbent_dict["feature"] = sample_features[incumbent]
                incumbent_dict["feature_norm"] = sample_norms[incumbent]
                incumbent_dict["expression"] = feature_space.phi[incumbent].expr
                incumbent_dict["Phi_value"] = self.M * (
                    incumbent_dict["value"] - self.alpha
                )
        return incumbent_dict, generated_sample_size

    def find_x_k(
        self, x_tilde_dict: dict, rho: np.ndarray, generated_sample_size: int
    ) -> dict:
        target_sample_size = generated_sample_size + 1
        incumbent_dict = x_tilde_dict
        while generated_sample_size < target_sample_size:
            P_incumbent = incumbent_dict["value"]  # P_k(x_k)
            interval_bound = 1 - incumbent_dict["Phi_value"] / (
                2 * self.M * np.linalg.norm(rho)
            )
            interval_probability = self.compute_probability(1, interval_bound)
            probability_tolerance = 1 - (
                (P_incumbent + 3 * self.alpha) / (2 * P_incumbent + 2 * self.alpha)
            )
            target_sample_size = self.compute_sample_size(
                interval_probability, probability_tolerance
            )
            feature_space = FeatureSpace(self.feature_inputs)
            sample_features_raw = np.array(
                [feature.value for feature in feature_space.phi]
            )
            sample_norms = np.linalg.norm(sample_features_raw, axis=1)
            sample_features = np.array(
                [
                    feature / norm
                    for feature, norm in zip(sample_features_raw, sample_norms)
                ]
            )
            generated_sample_size += len(sample_features)
            sample_values = np.abs(np.matmul(sample_features, rho))
            incumbent = np.argmax(sample_values)
            max_value = sample_values[incumbent]
            if max_value > incumbent_dict["value"]:
                incumbent_dict["value"] = max_value
                incumbent_dict["feature"] = sample_features[incumbent]
                incumbent_dict["feature_norm"] = sample_norms[incumbent]
                incumbent_dict["expression"] = feature_space.phi[incumbent].expr
                incumbent_dict["Phi_value"] = self.M * (
                    incumbent_dict["value"] - self.alpha
                )
        return incumbent_dict

    def solve_exact(self, tol: float, confidence: float) -> dict:
        u = np.array([])
        active_K_T = np.array([])
        active_norms = np.array([])
        active_expressions = np.array([])
        rho = self.rho(np.zeros((len(self.target), 1)), np.zeros(1))
        k = 0
        while True:
            # Test optimality
            x_tilde_dict, generated_sample_size = self.find_x_tilde(
                rho, tol, 1 - confidence
            )
            if x_tilde_dict["value"] < self.alpha / np.linalg.norm(rho):
                # optimality reached
                break

            # Update the iterate
            x_dict = self.find_x_k(x_tilde_dict, rho, generated_sample_size)
            eta_k = 8 / (k + 7)
            if x_dict["expression"] in active_expressions:
                index = np.where(active_expressions == x_dict["expressions"])[0][0]
                v = (
                    self.M
                    * np.sign(np.dot(rho, x_dict["feature"]))
                    * np.eye(1, len(u), index)[0]
                )
            else:
                if len(active_K_T):
                    active_K_T = np.vstack((active_K_T, x_dict["feature"]))
                else:
                    active_K_T = np.array([x_dict["feature"]])
                active_norms = np.append(active_norms, x_dict["feature_norm"])
                active_expressions = np.append(active_expressions, x_dict["expression"])
                u = np.append(u, np.array([0]))
                v = (
                    self.M
                    * np.sign(np.dot(rho, x_dict["feature"]))
                    * np.eye(1, len(u), len(u) - 1)[0]
                )
            u = (1 - eta_k) * u + eta_k * v

            # Low-dimensional optimization
            ssn = SSN(K=active_K_T.T, alpha=self.alpha, target=self.target, M=self.M)
            u_raw = ssn.solve(tol=self.machine_precision, u_0=u)
            u_to_keep = np.where(np.abs(u_raw) >= self.machine_precision)[0]
            u = u_raw[u_to_keep]
            active_K_T = active_K_T[u_to_keep]
            active_norms = active_norms[u_to_keep]
            active_expressions = active_expressions[u_to_keep]
            rho = self.rho(active_K_T.T, u)

            k += 1
            logging.info(f"{k}: Phi_gap {x_dict["Phi_value"]/self.M:.3E}, alternative_beta: {self.apha / np.linalg.norm(rho)}, support {len(u)}")
        logging.info(
            f"LGCG converged in {k} iterations to tolerance {tol:.3E} with confidence {confidence} and final sparsity of {len(u)}"
        )
        # Rescale the solution
        for ind, pos in enumerate(active_norms):
            u[ind] /= pos
        u = u * self.target_norm

        return {"u": u, "support": len(u), "expressions": active_expressions}
