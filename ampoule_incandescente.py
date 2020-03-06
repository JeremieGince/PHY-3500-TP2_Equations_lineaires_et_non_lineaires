import numpy as np


class MathFunction:

    """
    Class MathFunction permet la manipulation de fonction mathématique.
    """

    gaussxqValuesForN = dict()

    def __init__(self, func):
        """
        Constructeur de la class MathFunction.
        :param func: Fontion mathématique. (Function or lambda expression)
        """
        self.function = func

    def __call__(self, *args):
        """
        Call de la fonction mathématique courante.
        :param args: arguments de la fonction courante.
        :return: return de self.function(*args)
        """
        return self.function(*args)

    def gaussxq(self, N: int) -> (float, float):
        """
        A faire

        :param N:
        :return:
        """
        # Approximation initial des racines des polynomes de Legendre
        a = np.linspace(3, 4 * N - 1, N)/(4 * N + 2)
        x = np.cos(np.pi * a + 1 / (8 * (N**2) * np.tan(a)))

        # On trouve les racines avec la méthode de newton
        epsilon = 1e-15
        delta = 1.0
        while delta > epsilon:
            p0 = np.ones(N, float)
            p1 = np.copy(x)
            for k in range(1, N):
                p0, p1 = p1, ((2 * k + 1) * x * p1 - k*p0) /(k + 1)

            dp = (N + 1) * (p0 - x * p1) / (1 - x**2)
            dx = p1 / dp
            x -= dx
            delta = max(abs(dx))

        # Calcul des poids
        w = 2 * ((N + 1) ** 2) / ((N ** 2) * (1 - x ** 2) * (dp ** 2))

        return x, w

    def gaussian_quadrature_integration(self, **kwargs) -> float:
        # P.170 Computational physics
        assert "bounds" in kwargs and "N" in kwargs
        bounds = kwargs["bounds"]
        N = kwargs["N"]
        [a, b] = bounds

        # Calculate the sample points ans weights, then map them
        # to the required integration domain
        if N in self.gaussxqValuesForN:
            x, w = self.gaussxqValuesForN[N]
        else:
            x, w = self.gaussxq(N)
            self.gaussxqValuesForN[N] = (x, w)

        xp = 0.5 * (b - a) * x + 0.5 * (b + a)
        wp = 0.5 * (b - a) * w

        # perform the integration
        s = np.sum(wp * self.function(xp))
        return s

    def gaussian_quad_find_N(self, **kwargs) -> (int, float):
        import warnings
        assert "bounds" in kwargs
        bounds = kwargs["bounds"]
        error_target = kwargs.get("error_target", 1e-9)
        max_iteration = kwargs.get("max_iteration", 1_000)

        memory: dict = {
            "I_i": self.gaussian_quadrature_integration(bounds=bounds, N=1),
            "I_i+1": self.gaussian_quadrature_integration(bounds=bounds, N=2)
        }

        bestN: int = 2

        error: float = np.inf
        itr: int = 1
        for i in range(1, max_iteration + 1):
            error = np.abs(memory["I_i+1"] - memory["I_i"])
            if error <= error_target:
                itr = i
                break

            bestN = 2 ** (i + 1)

            memory["I_i"] = memory["I_i+1"]
            memory["I_i+1"] = self.gaussian_quadrature_integration(bounds=bounds, N=bestN)

        if itr >= max_iteration + 1:
            warnings.warn("Max iteration has been reach for gaussian_quad_find_N")

        return bestN, error

    def adaptative_gaussian_quadrature(self, **kwargs) -> (float, float):
        assert "bounds" in kwargs
        N, error = self.gaussian_quad_find_N(**kwargs)
        kwargs["N"] = N
        return self.gaussian_quadrature_integration(**kwargs), error


if __name__ == "__main__":

    def fonction(x):
        return x

    math = MathFunction(fonction)
    print(type(math.gaussian_quadrature_integration(bounds=[2, 3], N=100)))



