# first line: 89
        @memory.cache
        def _jac(x):
            g = self.jac(x, self.cov)
            if self.normalize:
                g = g/la.norm(g, np.inf)
            return g
