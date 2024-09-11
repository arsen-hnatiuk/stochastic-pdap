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
    # An implementation of the stochastic sampling approach to finite SPDAP

    def __init__(
        self,
        alpha: float,
        target: np.ndarray,
        feature_generator: Callable,
        # test_power: float = 0.05,
        # parameter_increase: float = 0.1,
    ) -> None:
        self.target_norm = np.linalg.norm(target)
        self.target = target / self.target_norm
        self.alpha = alpha
        self.feature_generator = feature_generator
        # self.test_power = test_power
        # self.parameter_increase = parameter_increase
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
            if size > 1e6:
                return 1e6
        return size

    def generate_sample_and_determine_incumbent(
        self, running_dict: dict, criterion: float, rho: np.ndarray
    ) -> tuple:
        incumbent_dict = {}

        # Generate new sample
        feature_space = self.feature_generator()
        sample_features_raw = np.array([feature.value for feature in feature_space.phi])
        sample_norms = np.linalg.norm(sample_features_raw, axis=1)
        sample_features = np.array(
            [feature / norm for feature, norm in zip(sample_features_raw, sample_norms)]
        )
        sample_values = np.abs(np.matmul(sample_features, rho))
        sample_expressions = np.array([feature.expr for feature in feature_space.phi])

        # Add sample to running dict
        if len(running_dict["expressions"]):
            running_dict["K_T"] = np.vstack((running_dict["K_T"], sample_features))
            running_dict["norms"] = np.append(running_dict["norms"], sample_norms)
            running_dict["expressions"] = np.append(
                running_dict["expressions"], sample_expressions
            )
            running_dict["values"] = np.append(running_dict["values"], sample_values)
        else:
            running_dict["K_T"] = sample_features
            running_dict["norms"] = sample_norms
            running_dict["expressions"] = sample_expressions
            running_dict["values"] = sample_values

        # Determine incumbent (best feature so far)
        incumbent = np.argmax(sample_values)
        max_value = sample_values[incumbent]
        if max_value > criterion:
            incumbent_dict["value"] = max_value
            incumbent_dict["feature"] = sample_features[incumbent]
            incumbent_dict["feature_norm"] = sample_norms[incumbent]
            incumbent_dict["expression"] = feature_space.phi[incumbent].expr
            incumbent_dict["Phi_value"] = self.M * (max_value - self.alpha)

        return incumbent_dict, running_dict

    def find_x_tilde(
        self,
        rho: np.ndarray,
        tol: float,
        probability_tolerance: float,
        running_dict: dict,
    ) -> tuple:
        # Check best existing feature
        if len(running_dict["expressions"]):
            incumbent = np.argmax(running_dict["values"])
            incumbent_dict = {
                "value": running_dict["values"][incumbent],
                "feature": running_dict["K_T"][incumbent],
                "feature_norm": running_dict["norms"][incumbent],
                "expression": running_dict["expressions"][incumbent],
                "Phi_value": self.M * (running_dict["values"][incumbent] - self.alpha),
            }
        else:
            incumbent_dict = {"value": -1}

        # Build test parameters
        null_beta = self.alpha * (1 + tol) / np.linalg.norm(rho)
        if null_beta >= 1:
            # The tolerance has been reached
            return incumbent_dict, running_dict
        alternative_beta = self.alpha / np.linalg.norm(rho)
        interval_probability = self.compute_probability(null_beta, alternative_beta)
        sample_size = self.compute_sample_size(
            interval_probability, probability_tolerance
        )
        logging.info(f"Sample size for x_tilde: {sample_size}")
        while (
            len(running_dict["expressions"]) < sample_size
            and incumbent_dict["value"] <= self.alpha
        ):
            incumbent_dict_tentative, running_dict = (
                self.generate_sample_and_determine_incumbent(
                    running_dict, alternative_beta, rho
                )
            )
            logging.info(
                f"Generated sample so far: {len(running_dict['expressions'])} with best value {incumbent_dict['value']}"
            )
            if len(incumbent_dict_tentative):
                incumbent_dict = incumbent_dict_tentative
        return incumbent_dict, running_dict

    def find_x_k(
        self, x_tilde_dict: dict, rho: np.ndarray, running_dict: dict
    ) -> tuple:
        # Generate needed sample size
        incumbent_dict = x_tilde_dict
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
        logging.info(f"Target sample size for x_k: {target_sample_size}")
        while len(running_dict["expressions"]) < target_sample_size:
            incumbent_dict_tentative, running_dict = (
                self.generate_sample_and_determine_incumbent(
                    running_dict, incumbent_dict["value"], rho
                )
            )
            logging.info(
                f"Generated sample so far: {len(running_dict['expressions'])} with best value {incumbent_dict['value']}"
            )
            if len(incumbent_dict_tentative):
                incumbent_dict = incumbent_dict_tentative

            # Compute new needed sample size
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
            logging.info(f"Target sample size for x_k: {target_sample_size}")
        return incumbent_dict, running_dict

    def solve_exact(self, tol: float, confidence: float) -> dict:
        u = np.array([])
        active_dict = {
            "K_T": np.array([]),
            "norms": np.array([]),
            "expressions": np.array([]),
        }
        running_dict = {
            "K_T": np.array([]),
            "norms": np.array([]),
            "expressions": np.array([]),
            "values": np.array([]),
        }
        rho = self.rho(np.zeros((len(self.target), 1)), np.zeros(1))
        k = 0
        while True:
            # Test optimality
            x_tilde_dict, running_dict = self.find_x_tilde(
                rho, tol, 1 - confidence, running_dict
            )
            if x_tilde_dict["value"] < self.alpha:
                # optimality reached
                break

            # Update the iterate
            x_dict, running_dict = self.find_x_k(x_tilde_dict, rho, running_dict)
            eta_k = 8 / (k + 7)
            if x_dict["expression"] in active_dict["expressions"]:
                index = np.where(active_dict["expressions"] == x_dict["expression"])[0][
                    0
                ]
                v = (
                    self.M
                    * np.sign(np.dot(rho, x_dict["feature"]))
                    * np.eye(1, len(u), index)[0]
                )
            else:
                if len(active_dict["K_T"]):
                    active_dict["K_T"] = np.vstack(
                        (active_dict["K_T"], x_dict["feature"])
                    )
                else:
                    active_dict["K_T"] = np.array([x_dict["feature"]])
                active_dict["norms"] = np.append(
                    active_dict["norms"], x_dict["feature_norm"]
                )
                active_dict["expressions"] = np.append(
                    active_dict["expressions"], x_dict["expression"]
                )
                u = np.append(u, np.array([0]))
                v = (
                    self.M
                    * np.sign(np.dot(rho, x_dict["feature"]))
                    * np.eye(1, len(u), len(u) - 1)[0]
                )
            u = (1 - eta_k) * u + eta_k * v

            # Low-dimensional optimization
            ssn = SSN(
                K=active_dict["K_T"].T, alpha=self.alpha, target=self.target, M=self.M
            )
            u_raw = ssn.solve(tol=self.machine_precision, u_0=u)
            u_to_keep = np.where(np.abs(u_raw) >= self.machine_precision)[0]
            u = u_raw[u_to_keep]
            active_dict["K_T"] = active_dict["K_T"][u_to_keep]
            active_dict["norms"] = active_dict["norms"][u_to_keep]
            active_dict["expressions"] = active_dict["expressions"][u_to_keep]
            rho = self.rho(active_dict["K_T"].T, u)
            running_dict["values"] = np.abs(np.matmul(running_dict["K_T"], rho))

            k += 1
            logging.info(
                f"{k}: Phi_gap {x_dict['Phi_value']/self.M:.3E}, alternative_beta: {self.alpha / np.linalg.norm(rho)}, support {active_dict['expressions']}"
            )
        logging.info(
            f"LGCG converged in {k} iterations to tolerance {tol:.3E} with confidence {confidence} and final sparsity of {len(u)}"
        )
        # Rescale the solution
        for ind, pos in enumerate(active_dict["norms"]):
            u[ind] /= pos
        u = u * self.target_norm

        return {"u": u, "support": len(u), "expressions": active_dict["expressions"]}


if __name__ == "__main__":
    ops = [
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
    ]
    feature_inputs = import_dataframe.create_inputs(
        df="data/thermal_conductivity_data.csv",
        max_rung=3,
        max_param_depth=0,
        prop_key="log kappa_L",
        calc_type="regression",
        n_sis_select=10,
        allowed_ops=ops,
        n_rung_generate=0,
        n_rung_store=-1,
        allowed_param_ops=[],
        global_param_opt=False,
        reparam_residual=False,
    )
    feature_generator = lambda: FeatureSpace(feature_inputs)
    target = pd.read_csv("data/thermal_conductivity_data.csv")["log kappa_L"].to_numpy()
    exp = SPDAP_Finite(0.1, target, feature_generator)
    exp.solve_exact(tol=0.001, confidence=0.999)
