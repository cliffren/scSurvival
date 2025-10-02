import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

class Logger:
    def __init__(self, save_path=None):
        self.logger = {}
        self.save_path = save_path

    def log(self, key, value):
        if key in self.logger:
            self.logger[key].append(value)
        else:
            self.logger[key] = [value]
    
    def log_dict(self, log_dict):
        for key, value in log_dict.items():
            self.log(key, value)

    def get_logs(self):
        return self.logger

    def get_logs_df(self):
        df = pd.DataFrame(self.logger)
        if self.save_path:
            df.to_csv(self.save_path, index=False)
        return df


# ---- 小工具函数 ----
def r_to_pandas(name: str):
    """把 R 端的 data.frame/matrix 取回为 pandas.DataFrame"""
    robj = ro.globalenv[name]
    with localconverter(ro.default_converter + pandas2ri.converter):
        try:
            return ro.conversion.rpy2py(robj)  # 若是 data.frame，会自动转 pandas.DataFrame
        except Exception:
            pass
    # 若是 matrix（有时不会直接转 DF），手动转
    # 用 R 把 matrix 转 data.frame 再取回
    with localconverter(ro.default_converter + pandas2ri.converter):
        robj_df = ro.r(f"as.data.frame({name})")
        return ro.conversion.rpy2py(robj_df)

def r_list_to_pydict_df(name: str, rownames_as_index: bool = True):
    """把 R 的 list（元素为 data.frame 或 matrix）转为 Python 的 dict[str, pandas.DataFrame]"""
    r_list = ro.globalenv[name]
    out = {}
    for k, v in zip(r_list.names, r_list):
        key = str(k)
        # 尝试直接转成 pandas.DataFrame
        with localconverter(ro.default_converter + pandas2ri.converter):
            try:
                df = ro.conversion.rpy2py(v)
            except Exception:
                # 如果是 matrix 等，先在 R 里转成 data.frame
                df = ro.conversion.rpy2py(ro.r("as.data.frame")(v))
        # 可选：把 R 的 rownames 放到 DataFrame 索引
        if rownames_as_index:
            # rpy2 转换时通常会把 rownames 放在 index，如果没有，就从 R 侧拿 rownames
            if df.index.dtype.kind in ("i","u") and (df.index == range(len(df))).all():
                rn = list(ro.r("rownames")(v))
                if rn and len(rn) == len(df):
                    df.index = rn
        out[key] = df
    return out
