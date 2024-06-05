import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import ipywidgets as widgets

from IPython import display



def undo_topcode(df, topcoded_age, l_age, r_age):
    ret = df.copy(deep=True)
    if "Race" in ret.columns:
        pass
    else:
        for sex in ["M", "F"]:
            total_ct = ret.loc[(sex, topcoded_age)]["Population"]

            new_row_ct = r_age - l_age + 1

            #divides age uniformly 
            ind_ct = int(float(total_ct) / float(new_row_ct))

            new_rows = pd.DataFrame(
                {
                    "Sex" : [sex for _ in range(new_row_ct)], 
                    "Age" : [i for i in range(l_age, r_age + 1)],
                    "Population" : [ind_ct for _ in range(new_row_ct)]
                }, index=[(sex, age) for age in range(l_age, r_age+1)]
            )

            ret = pd.concat([ret.drop((sex, topcoded_age)), new_rows])

        return ret.sort_index()
    
def undo_topcode_columns(df, topcoded_age, l_age, r_age, rate_statistic=False):
    ret = df.copy(deep=True)
    if "Race" in ret.columns:
        raise "undo_topcode_columns failed"
    else:
        new_cols = {f"{age}" : [] for age in range(l_age, r_age+1)}

        for i, row in ret.iterrows():
            total_ct = row[f"{topcoded_age}"]
            new_col_ct = r_age - l_age + 1

            ind_ct = total_ct if rate_statistic else int(float(total_ct) / float(new_col_ct))

            for k,v in new_cols.items():
                v.append(ind_ct)

        ret = pd.concat([ret.drop(f"{topcoded_age}", axis=1).reset_index(drop=True), pd.DataFrame(new_cols).reset_index(drop=True)], axis=1)

        return ret.sort_index()
    
def project_out(init_pop_df, fert_df, migr_df, mort_df, start_year, end_year):
    male_sex_ratio = 0.5169
    female_sex_ratio = 1.0 - male_sex_ratio

    sex_ratio_dict = {"M":male_sex_ratio, "F":female_sex_ratio}

    age_range = (0, 100)
    fertile_range = (14, 54)

    yoy_pops = []

    yoy_pops.append(init_pop_df["Population"].sum())

    curr_pop_df = init_pop_df.copy(deep=True)

    for year in range(start_year+1, end_year+1):
        curr_pop_df = curr_pop_df.set_index(["Sex", "Age"], drop=False)
        
        new_dfs = []
        for sex in ["M", "F"]:
            pop_series = curr_pop_df[curr_pop_df["Sex"] == sex]["Population"].to_numpy()[:-1]

            mort_series = mort_df.loc[(year, sex)].to_numpy()[:-1]
            
            after_mort_series = pop_series * (1.0 - mort_series)

            births = 0

            for maternal_age in range(fertile_range[0], fertile_range[1] + 1):
                births += int(curr_pop_df.loc[("F", maternal_age)]["Population"] * fert_df.loc[year][f"{maternal_age}"] * sex_ratio_dict[sex]) 

            after_birth_series = np.concatenate((np.array([int(births)]), after_mort_series))

            migr_series = migr_df.loc[(year, sex)].to_numpy()

            after_migr_series = after_birth_series + migr_series

            sex_series = [sex for i in range(age_range[0], age_range[1] + 1)]
            age_series = [i for i in range(age_range[0], age_range[1] + 1)]

            new_dfs.append(pd.DataFrame({"Age" : age_series, "Sex" : sex_series, "Population":after_migr_series}))
        curr_pop_df = pd.concat(new_dfs, ignore_index=True)
        yoy_pops.append(curr_pop_df["Population"].sum())

    return yoy_pops

class SimpleProjection:
    def __init__(self, label, years, pops, future=True):
        self.label = label 
        self.years = years 
        self.pops = pops 
        self.future = future

    def has_widget(self):
        return False

class MultiProjection:
    def __init__(self, label, years, pop_fn, widget, future=True):
        self.label = label 
        self.years = years 
        self.pops = pop_fn(widget.value) 
        self.pop_fn = pop_fn
        self.future = future
        self.widget = widget

        self.grapher = None

    def bind(self, grapher):
        self.grapher = grapher 

        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.pops = self.pop_fn(self.widget.value)

                if self.grapher is not None:
                    self.grapher.show()

        self.widget.observe(on_change)


    def has_widget(self):
        return True

class Grapher:
    def __init__(self, projections = None, bounds=None) -> None:
        self.plot_out = widgets.Output()
        
        self.widget_list = [self.plot_out]

        self.projections = []

        if projections is not None:
            self.projections = projections

            for projection in self.projections:
                if projection.has_widget():
                    self.widget_list.append(projection.widget)

                    projection.bind(self)

        self.full_out = widgets.VBox(self.widget_list)
        display.display(self.full_out)
    

    def show(self):
        plt.clf()

        plt.title("US Population (millions)")

        plt.xlabel("Year")

        for projection in self.projections:

            plt.plot(projection.years, projection.pops, linestyle="--" if projection.future else "-", label=projection.label)

        # Function to format the y-axis
        def millions_formatter(x, pos):
            return '%1.0f' % (x * 1e-6)

        # Set the formatter
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(millions_formatter))

        plt.grid()

        plt.legend()
        
        self.plot_out.clear_output(wait=True)
        with self.plot_out:
            plt.show()

        # display.display(self.full_out)
        # display.display(plt.gcf())
        # for projection in self.projections:
        #     if projection.has_widget():
        #         display.display(projection.widget)