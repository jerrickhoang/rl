class LQR(object):

    def __init__(self, env, T, F, f, C, c):
        self._u_dim = env.action_space.shape[0]
        self._x_dim = env.observation_space.shape[0]
        self._T = T
        self._F = [F] * T
        self._f = [f] * T
        self._C = [C] * T
        self._c = [c] * T

  def _partition(self, m, x_dim, u_dim):
      return m[x_dim:, x_dim:], m[x_dim:, :x_dim], m[:x_dim, x_dim:], m[:x_dim, :x_dim]

  def backward_pass(self, T):
      Ks = [0] * T  # [u, x]
      ks = [0] * T  # [u, 1]
      Vs = [0] * T  # [x, x]
      vs = [0] * T  # [x, 1]
      Qs = [0] * T  # [x + u, x + u]
      qs = [0] * T  # [x + u, 1]
      C = self._C   # [x + u, x + u]
      F = self._F   # [x, x + u] 
      f = self._f   # [x, 1]
      c = self._c   # [x + u, 1]

    for t in reversed(range(T)):
        if t == T - 1:
            C_uu, C_ux, C_xu, C_xx = self._partition(self._C[T-1], self._x_dim, self._u_dim)
            C_uu_inv = np.linalg.inv(C_uu)
            c_u, c_x = self._c[T-1][self._x_dim:], self._c[T-1][:self._x_dim]
            Ks[T - 1] = -np.dot(C_uu_inv, (0.5 * (C_xu.T + C_ux)))
            ks[T - 1] = -C_uu_inv.dot(c_u)
            Vs[T - 1] = C_xx + C_xu.dot(Ks[T-1]) + Ks[T-1].T.dot(C_ux) + Ks[T-1].T.dot(C_uu).dot(Ks[T-1])
            vs[T - 1] = c_x + C_xu.dot(ks[T-1]) + Ks[T-1].T.dot(c_u) + Ks[T-1].T.dot(C_uu).dot(ks[T-1])
            continue
        Qs[t] = C[t] + F[t].T.dot(Vs[t+1]).dot(F[t])
        qs[t] = c[t] + F[t].T.dot(Vs[t+1]).dot(f[t]) + F[t].T.dot(vs[t+1])
        q_u, q_x = qs[t][self._x_dim:], qs[t][:self._x_dim]
        Q_uu, Q_ux, Q_xu, Q_xx = self._partition(Qs[t], self._x_dim, self._u_dim)
        Q_uu_inv = np.linalg.inv(Q_uu)
        Ks[t] = -Q_uu_inv.dot(Q_ux)
        ks[t] = -Q_uu_inv.dot(q_u)
        Vs[t] = Q_xx + Q_xu.dot(Ks[t]) + Ks[t].T.dot(Q_ux)+Ks[t].T.dot(Q_uu).dot(Ks[t])
        vs[t] = q_x + Q_xu.dot(ks[t]) + Ks[t].T.dot(q_u) + Ks[t].T.dot(Q_uu).dot(ks[t])
    return Ks, ks

def forward_pass(self, state, Ks, ks):
    return Ks[0] * state + ks[0]

def plan(self, state):
    Ks, ks = self.backward_pass(self._T)
    u = self.forward_pass(state, Ks, ks)
    return u
