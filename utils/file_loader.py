import os
import numpy as np
import lightgbm as lgb


class FileLoader(object):

    def __init__(self, directory, prefix, config_file='train.conf'):
        try:
            directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), directory)
        except:
            directory = os.getcwd()

        self.directory = directory
        self.prefix = prefix
        self.params = {'gpu_use_dp': False}
        with open(os.path.join(directory, config_file), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = [token.strip() for token in line.split('=')]
                    if 'early_stopping' not in key:  # disable early_stopping
                        self.params[key] = value

    def load_dataset(self, suffix, is_sparse=False):
        filename = self.path(suffix)
        if is_sparse:
            X, Y = load_svmlight_file(filename, dtype=np.float64, zero_based=True)
            return X, Y, filename
        else:
            mat = np.loadtxt(filename, dtype=np.float64)
            return mat[:, 1:], mat[:, 0], filename

    def load_field(self, suffix):
        return np.loadtxt(os.path.join(self.directory, self.prefix + suffix))

    def load_cpp_result(self, result_file='LightGBM_predict_result.txt'):
        return np.loadtxt(os.path.join(self.directory, result_file))

    def train_predict_check(self, lgb_train, X_test, X_test_fn, sk_pred):
        gbm = lgb.train(self.params, lgb_train)
        y_pred = gbm.predict(X_test)
        cpp_pred = gbm.predict(X_test_fn)
        np.testing.assert_array_almost_equal(y_pred, cpp_pred, decimal=5)
        np.testing.assert_array_almost_equal(y_pred, sk_pred, decimal=5)

    def file_load_check(self, lgb_train, name):
        lgb_train_f = lgb.Dataset(self.path(name), params=self.params).construct()
        for f in ('num_data', 'num_feature', 'get_label', 'get_weight', 'get_init_score', 'get_group'):
            a = getattr(lgb_train, f)()
            b = getattr(lgb_train_f, f)()
            if a is None and b is None:
                pass
            elif a is None:
                assert np.all(b == 1), f
            elif isinstance(b, (list, np.ndarray)):
                np.testing.assert_array_almost_equal(a, b)
            else:
                assert a == b, f

    def path(self, suffix):
        return os.path.join(self.directory, self.prefix + suffix)
