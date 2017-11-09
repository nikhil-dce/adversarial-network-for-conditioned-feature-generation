class Config:

    def __init__(self, batch_size, x_dim, attr_dim, z_dim, gh_dim, dh_dim, lr, g_steps, d_steps):

        self.batch_size = batch_size
        self.x_dim = x_dim
        self.attr_dim = attr_dim
        self.z_dim = z_dim
        self.gh_dim = gh_dim
        self.dh_dim = dh_dim
        self.lr = lr
        self.g_steps = g_steps
        self.d_steps = d_steps

    def print_settings(self):

        print 'Batch Size: %d' % self.batch_size
        print 'X_Dim: %d' % self.x_dim
        print 'Attr_Dim: %d' % self.attr_dim
        print 'Z_dim: %d' % self.z_dim
        print 'DH_dim: %d' % self.dh_dim
        print 'GH_dim: %d' % self.gh_dim
        print 'learning_rate: %d' % self.lr
        print 'g_steps: %d' % self.g_steps
        print 'd_steps: %d' % self.d_steps
