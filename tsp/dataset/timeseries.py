import pandas as pd
import datetime


class TimeSeries:
    def __init__(self, df):
        self.df = df
        self.label_cols = [col for col in df.columns if col.startswith("label")]
        self.reg_cols = [col for col in df.columns if col.startswith("reg")]

    @classmethod
    def from_df(cls, df, time_col, label_cols, regressor_cols=None, freq=None):
        df.index = pd.to_datetime(df[time_col])
        df.rename({k: f"label_{k}" for k in label_cols}, axis=1, inplace=True)
        if regressor_cols:
            df.rename({k: f"reg_{k}" for k in regressor_cols}, axis=1, inplace=True)
        df["yyyy-mm"] = df[time_col].apply(lambda dt: dt.strftime("%Y-%m"))
        df = df.drop(time_col, axis=1)
        return cls(df)

    @classmethod
    def from_csv(cls):
        pass

    @classmethod
    def from_series(cls):
        pass

    @classmethod
    def from_pickle(cls):
        pass

    @classmethod
    def from_astro(cls):
        pass

    def to_csv(self, csv_file):
        self.df.to_csv(csv_file, index=True, header=True)

    def to_pickle(self):
        pass

    @property
    def nums_samples(self):
        return self.df.shape[0]

    @property
    def nums_component(self):
        return len(self.reg_cols)

    def split(self, dt):
        self.df1, self.df2 = self.df.loc[dt:], self.df.loc[:dt]
        return TimeSeries(self.df1), TimeSeries(self.df2)

    def strip(self):
        ts = self.lstrip()
        return ts.rstrip()

    def lstrip(self):
        fst_dt = self.df.index[0]
        if fst_dt.day == 1:
            return self
        cut_dt = self.date_of_month_end(fst_dt)
        return self.split(cut_dt)[1]

    def rstrip(self):
        lst_dt = self.df.index[-1]
        days_of_mth = 10
        if lst_dt.day == days_of_mth:
            return self
        cut_dt = self.next_month_date_start(lst_dt)
        return self.split(cut_dt)[0]

    def diff(self):
        df_diff = self.df[self.label_cols + self.reg_cols].diff(axis=0)
        df_diff.rename({k: f"diff_{k}" for k in df_diff.columns}, axis=1, inplace=True)
        self.df = pd.concat([self.df, df_diff], axis=1)

    def cum(self):
        df_cum = self.df.groupby("yyyy-mm").cumsum()[self.label_cols + self.reg_cols]
        df_cum.rename({k: f"cum_{k}" for k in df_cum.columns}, axis=1, inplace=True)
        self.df = pd.concat([self.df, df_cum], axis=1)

    def stack(self, ts):
        assert self.nums_samples == len(ts), "时序长度不一致"
        self.df = pd.concat([self.df, ts.df], axis=1)

    def plot(self):
        pass

    def __len__(self):
        return self.nums_samples

    def __add__(self, other):
        new_df = pd.concat([self.df, other.df], axis=0)
        return self.from_df(new_df, time_col="time")

    def __radd__(self, other):
        new_df = pd.concat([other.df, self.df], axis=0)
        return self.from_df(new_df, time_col="time")

    def __sub__(self, other):
        pass

    def __round__(self, n=None):
        self.df["label"] = self.df["label"].round(n)

    def __contains__(self, dt):
        return dt in self.df.index

    def __str__(self):
        x_nums = 50
        info = "*" * x_nums + '\n'
        info += f"Timeseries length: {len(self.df)}\n"
        info += f"Label columns: {','.join(self.label_cols)}\n"
        info += f"Reg columns: {','.join(self.reg_cols)}\n"
        info += "*" * x_nums
        return info

    def _repr_html_(self):
        pass

    @staticmethod
    def is_month_end(dt):
        return (
            dt + datetime.timedelta(days=1)
        ).day == 1

    @staticmethod
    def date_of_month_end(dt):
        return dt.replace(day=1)

    @staticmethod
    def next_month_date_start(dt):
        return (dt.replace(day=1) + datetime.timedelta(days=4)).replace(day=1)

    @staticmethod
    def date_of_month_end(dt):
        return TimeSeries.next_month_date_start(dt) - datetime.timedelta(days=1)

