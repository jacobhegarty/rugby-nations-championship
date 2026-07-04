
######################################imports
###See notebook for details on models.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
###################################################################################################################
#                                     PLOT THEME - COURESY OF CLAUDE                                              #
###################################################################################################################
plt.rcParams.update({
    "figure.facecolor": "#0A0E17",
    "axes.facecolor": "#0F1524",
    "savefig.facecolor": "#0A0E17",
    "text.color": "#E8F1FF",
    "axes.edgecolor": "#1C2740",
    "axes.labelcolor": "#E8F1FF",
    "xtick.color": "#5E7A99",
    "ytick.color": "#5E7A99",
    "grid.color": "#1C2740",
    "font.family": "sans-serif",
    "font.sans-serif": ["Consolas", "DejaVu Sans Mono", "Arial"],
})
sns.set_theme(style="dark", rc=plt.rcParams)   # "dark", not "whitegrid"; called AFTER rcParams.update

BG_COLOR     = "#0A0E17"   # near-black background
PANEL_COLOR  = "#0F1524"   # slightly lighter panel
INK_COLOR    = "#E8F1FF"   # icy white-blue text
SUB_COLOR    = "#5E7A99"   # muted slate for ticks/subtitles
GRID_COLOR   = "#1C2740"   # faint gridlines

HOME_COLOR   = "#007489"   # orange - home team, always
AWAY_COLOR   = "#e8b663"   # blue - away team, always
DRAW_COLOR   = "#c0c0c0"   # slate grey - draws
ACCENT_COLOR = "white"     # sparing highlight colour
OBSERVED_COLOR = "#E8F1FF" # icy white - observed data overlays

WIN_COLOR    = "#00845c"   # green - good outcome / good position
LOSS_COLOR   = "#B5282F"   # red - bad outcome / bad position

# panel -> win colour, single-hue ramp (was fading teal->orange before, now consistent)
HEATMAP_CMAP   = LinearSegmentedColormap.from_list("win_glow", [PANEL_COLOR, "#0F5C40", WIN_COLOR])
# red (home always loses, P=0) -> white -> green (home always wins, P=1)
DIVERGING_CMAP = LinearSegmentedColormap.from_list("loss_win_div", [LOSS_COLOR, "white", WIN_COLOR])
# straight green -> red gradient: best finishing position -> worst
POSITION_CMAP = ListedColormap([
    "#0F172A",  # near-black navy
    "#1E3A8A",
    "#2563EB",
    "#60A5FA",
    "#93C5FD",
    "#DBEAFE",  # near-white
])
TEAM_CMAP = "cividis"

###################################################################################################################
#                                     MAIN RUGBY MODEL CLASS                                                      #
###################################################################################################################

class RugbyModel:

    # teams
    sixNations  = {"England", "Ireland", "France", "Scotland", "Wales", "Italy"}
    fourNations = {"All Blacks", "South Africa", "Australia", "Argentina"}
    focalTeams = {*sixNations,*fourNations}


    def __init__(self,  half_life=1.0, decayRate = 0.8,nSamples=10000, tune=1500, cores=4,focalTeams = focalTeams,target_accept = 0.99,contWeights = True):
        self.half_life = half_life
        self.nSamples = nSamples
        self.tune = tune
        self.cores = cores
        self.focalTeams = focalTeams
        self.target_accept = target_accept
        self.contWeights = contWeights
        self.decayRate = decayRate
    
    def fit(self, df):
        
        #all focal teams and the teams they play in data
        opponent_teams = set(df.loc[df["home_team"].isin(self.focalTeams), "away_team"]) | \
                        set(df.loc[df["away_team"].isin(self.focalTeams), "home_team"])
        self.opponent_teams = opponent_teams

        #every team that will need a parameter in the model (focal + opponents)
        all_model_teams = sorted(set(self.focalTeams) | opponent_teams)
        #map team name -> integer index for indexing into arrays
        team_to_idx = {t: i for i, t in enumerate(all_model_teams)}

        self.team_to_idx = team_to_idx
        self.n_teams     = len(team_to_idx)
        
        #filter data to only contain games with at least one focal team
        df = df.loc[
            df["home_team"].isin(self.focalTeams) | df["away_team"].isin(self.focalTeams)
        ].copy()

        self.fit_df = df
        
        #temporal weights
        if self.contWeights: 
            refDate = df["date"].max() 
            xi       = np.log(2) / (self.half_life * 365)
            days_old = (refDate - df["date"]).dt.days.values
            weights  = np.exp(-xi * days_old)
        else: # discrete dorp of for weights - experimental 
            refDate = pd.to_datetime("2027-01-01")# doing it on the year is pretty resonable - keeps 
            yearsOld = ((refDate - df["date"]).dt.days // 365).values
            weights = self.decayRate ** yearsOld 


        self._build( weights, df) #fit model 
        return self
    

    def eval_result(self, dfTest, plot=True):
        """Evaluate match-result (win/draw/loss) accuracy on a test set, optionally plotting a confusion matrix."""

        #clean test data
        dfTest = self._val_test(dfTest)
        if len(dfTest) == 0:
            return np.nan
                
        matches = []
        predicted_outcomes = []
        actual_outcomes = []

        for _, row in dfTest.iterrows():#loop test data
            #get indexes
            hi = self.team_to_idx[row["home_team"]]
            ai = self.team_to_idx[row["away_team"]]

            #simulate match, only need score
            sh, sa, *_ = self.predict_match(hi, ai)

            #get winner
            outcome_samples = np.sign(sh - sa)
            #count wins
            values, counts = np.unique(outcome_samples, return_counts=True)
            #mode winner 
            predicted_outcome = values[np.argmax(counts)]

            #who actually won
            actual_outcome = np.sign(row["home_score"] - row["away_score"])

            #was it correct?
            matches.append(predicted_outcome == actual_outcome)
            predicted_outcomes.append(predicted_outcome)
            actual_outcomes.append(actual_outcome)

        if plot: # plot confusion matrix
            label_map = {-1: "Away Win", 0: "Draw", 1: "Home Win"}
            labels = [-1, 0, 1]
            label_names = [label_map[l] for l in labels]

            #build the counts matrix (rows = actual, cols = predicted)
            cm = np.zeros((3, 3), dtype=int)
            for pred, actual in zip(predicted_outcomes, actual_outcomes):
                i = labels.index(actual)
                j = labels.index(pred)
                cm[i, j] += 1

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap=HEATMAP_CMAP,
                        xticklabels=label_names, yticklabels=label_names, ax=ax,
                        linewidths=0.5, linecolor="white", cbar_kws={"label": "count"})
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Result confusion matrix")
            plt.tight_layout()
            plt.show()
        
        #print accuracy
        print(f"Match result accuracy = {float(np.mean(matches))}")

        #returns accuracy
        return float(np.mean(matches))
    
    #internal function - plots actual vs predicted for component passed, filtered for a specific country if a country is provided
    def _eval_component_country(self, dfTest, component,  country=None,plot=True):
        #clean test data
        dfTest = self._val_test(dfTest)

        #restrict to games involving the given country if one was supplied
        if country is not None:
            mask = (dfTest["home_team"] == country) | (dfTest["away_team"] == country)
            dfTest = dfTest[mask]

        if len(dfTest) == 0:
            return np.nan

        errors = []
        errorsDiff = []
        preds = []
        actuals = []
        isCountry = []  # whether each pred/actual point belongs to `country`
        print(f"\n\n{component.capitalize()} level evaulation:\n")
        print(f"    Number of usable tests: {len(dfTest)}\n")

        tot = 0
        totPred = 0
        for _, row in dfTest.iterrows():
            #get indexes
            hi = self.team_to_idx[row["home_team"]]
            ai = self.team_to_idx[row["away_team"]]

            if component == "try":
                #predict match and pull out try counts
                _, _, th, ta, ph, pa, *_ = self.predict_match(hi, ai)
                pred_home = np.mean(th)
                pred_away = np.mean(ta)
                actual_home = row["home_tries"] + row["home_penalty_tries"]
                actual_away = row["away_tries"] + row["away_penalty_tries"]

            elif component == "penalty":
                #predict match and pull out penalty counts
                _, _, th, ta, ph, pa, *_ = self.predict_match(hi, ai)
                pred_home = np.mean(ph)
                pred_away = np.mean(pa)
                actual_home = row["home_penalties"]
                actual_away = row["away_penalties"]

            elif component == "conversion":
                #predict match and pull out conversion counts
                *_, ch, ca = self.predict_match(hi, ai)
                pred_home = np.mean(ch)
                pred_away = np.mean(ca)
                actual_home = row["home_conversions"]
                actual_away = row["away_conversions"]

            elif component == "score":
                #predict match and pull out final score
                sh, sa, *_ = self.predict_match(hi, ai)
                pred_home = np.mean(sh)
                pred_away = np.mean(sa)
                actual_home = row["home_score"]
                actual_away = row["away_score"]

            else:
                raise ValueError(f"Unknown component: {component}")

            #running totals for average print-out
            tot += actual_home + actual_away
            totPred += pred_home + actual_home

            #absolute error and signed differential error for this match
            errors.append((abs(pred_home - actual_home) + abs(pred_away - actual_away)/2))
            errorsDiff.append((pred_home - pred_away) - (actual_home- actual_away))
            preds.extend([pred_home, pred_away])
            actuals.extend([actual_home, actual_away])

            #track which points belong to the filtered country vs opponents
            if country is not None:
                isCountry.extend([row["home_team"] == country, row["away_team"] == country])
            else:
                isCountry.extend([True, True])

        print(f"    Average observed {component} = {tot/(len(dfTest)*2)} per team per game.")
        print(f"    Average predicted {component} = {totPred/(len(dfTest)*2)} per team per game.")

        print(f"    MAE {component} = {float(np.mean(errors))} per team per game.")
        print(f"    MAE {component} difference = {float(np.mean(np.abs(errorsDiff)))}")


        if plot:
            fig, (ax0, ax) = plt.subplots(1, 2, figsize=(12, 5))

            sns.histplot(np.abs(errorsDiff), ax=ax0, color=HOME_COLOR, edgecolor="white", kde=True)
            ax0.set_title("Distribtuion of errors")

            preds = np.array(preds)
            actuals = np.array(actuals)
            isCountry = np.array(isCountry)

            #scatter predicted vs actual, colouring the filtered country separately if given
            if country is not None:
                ax.scatter(actuals[isCountry], preds[isCountry], alpha=0.7,
                            color=HOME_COLOR, label=country, edgecolor="white", linewidth=0.3)
                ax.scatter(actuals[~isCountry], preds[~isCountry], alpha=0.5,
                            color=AWAY_COLOR, label="Opponents", edgecolor="white", linewidth=0.3)
                ax.legend()
            else:
                ax.scatter(actuals, preds, alpha=0.6, color=HOME_COLOR, edgecolor="white", linewidth=0.3)

            #draw the y=x reference line so perfect predictions are visible
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, "k--", alpha=0.5, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(f"Actual {component}")
            ax.set_ylabel(f"Predicted {component}")
            title = f"Predicted vs actual {component}"
            if country is not None:
                title += f" ({country})"
            ax.set_title(title)
            plt.tight_layout()
            plt.show()

        return float(np.mean(np.abs(errorsDiff)))

    #these methods wrap the above for each score component
    def eval_tries(self, dfTest,country=None):

        return self._eval_component_country(dfTest, "try",country)

    def eval_pens(self, dfTest,country = None):
        return self._eval_component_country(dfTest, "penalty",country)

    def eval_convs(self, dfTest,country = None):
        return self._eval_component_country(dfTest, "conversion",country)
    
    def eval_score(self, dfTest,country = None):
        return self._eval_component_country(dfTest, "score",country)


    #unpack data - makes it easier to track variables through model 
    def _unpack_data(self,df):
        #observed tries (including penalty tries) for home/away
        obs_tries_home = df["home_tries"].values + df["home_penalty_tries"].values 
        obs_tries_away = df["away_tries"].values + df["away_penalty_tries"].values 

        #observed penalty goals for home/away
        obs_pens_home = df["home_penalties"].values
        obs_pens_away = df["away_penalties"].values

        #observed conversions for home/away
        obs_conv_home = df["home_conversions"].values
        obs_conv_away = df["away_conversions"].values
                
        #observed final scores for home/away
        obs_points_home =  df["home_score"].values
        obs_points_away = df["away_score"].values

        
        #map team names to integer indices for this dataframe
        home_idx = df["home_team"].map(self.team_to_idx).values
        away_idx = df["away_team"].map(self.team_to_idx).values

        
        #observed drop goals for home/away
        obs_dgs_home =  df["home_drop_goals"].values
        obs_dgs_away = df["away_drop_goals"].values

        

        return home_idx,away_idx,obs_tries_home,obs_tries_away,obs_pens_home,obs_pens_away,obs_conv_home,obs_conv_away,obs_points_home,obs_points_away,obs_dgs_home,obs_dgs_away

    #removes any games in test set which have an unseen team or dont include a focal team
    def _val_test(self,dfTest):
        #at least one focal team
        dfTest = dfTest.loc[
            dfTest["home_team"].isin(self.focalTeams) | dfTest["away_team"].isin(self.focalTeams)
        ].copy()

        #no unseen teams. Note focal teams connected enoguh that all will also appear in opponent teams
        dfTest = dfTest.loc[
            dfTest["home_team"].isin(self.opponent_teams) & dfTest["away_team"].isin(self.opponent_teams)
        ].copy()
        return dfTest

    def plot_team_params(self, params=["att", "def_"], teams=None, ncols=2, figsize_per_plot=(6, 4)):
        """
        Plot the posterior distribution of each per-team parameter, one subplot
        per parameter, with each team's distribution overlaid in that subplot.

        Only plots parameters that (a) are attributes on self, (b) are arrays,
        and (c) have shape (n_samples, n_teams) - i.e. per-team.
        """
        candidate_params = params 

        #only keep parameters that actually exist on self and are per-team arrays
        available_params = []
        for p in candidate_params:
            val = getattr(self, p, None)
            if val is None:
                continue
            val = np.asarray(val)
            if val.ndim == 2 and val.shape[1] == self.n_teams:
                available_params.append(p)

        if not available_params:
            print("No per-team parameters found on this model.")
            return

        #default to focal teams, in a stable order
        teams_to_plot = teams or sorted(self.focalTeams, key=lambda t: self.team_to_idx[t])
        team_indices = [self.team_to_idx[t] for t in teams_to_plot]

        n_params = len(available_params)
        ncols = min(ncols, n_params)
        nrows = int(np.ceil(n_params / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
            squeeze=False
        )
        axes = axes.flatten()

        spacing = 1.0
        paramTitles = {
            "att": "Attacking Strength (Tries)",
            "def_": "Defensive Strength (Tries)",
            "dis_clean": "Avoiding Penalties",
            "dis_force": "Scoring Penalties",
            "p_conv": "Conversion Success",
        }
        for ax_idx, param in enumerate(available_params):
            ax = axes[ax_idx]
            val = np.asarray(getattr(self, param))

            for i, (team, t_idx) in enumerate(zip(teams_to_plot, team_indices)):
                x = val[:, t_idx]

                #kde of this team's posterior for this parameter
                kde = gaussian_kde(x)

                xmin, xmax = np.min(x), np.max(x)
                xgrid = np.linspace(xmin, xmax, 300)

                y = kde(xgrid)

                # scale all curves to similar height
                y = y / y.max() * 0.8

                ax.plot(xgrid, y + i * spacing, lw=2)
                ax.fill_between(xgrid, i * spacing, y + i * spacing, alpha=0.25)

            ax.set_yticks(np.arange(len(teams_to_plot)) * spacing)
            ax.set_yticklabels(teams_to_plot)

            ax.set_title(paramTitles.get(param, param))
            ax.set_xlabel("Value")
            ax.set_ylabel("Team")

        # hide any unused axes
        for ax_idx in range(n_params, len(axes)):
            axes[ax_idx].axis("off")

        # single shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes[:n_params]:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        plt.tight_layout()
        plt.show()

        return available_params

    def _observed_team_stats(self, team, opponents=None):
        """
        Pull observed tries, points, penalties for a team across all matches
        in self.fit_df (home or away), optionally restricted to a set of opponents.
        """
        df = self.fit_df

        #matches where this team played at home / away
        home_mask = df["home_team"] == team
        away_mask = df["away_team"] == team

        #optionally restrict to matches against a given opponent pool
        if opponents is not None:
            home_mask &= df["away_team"].isin(opponents)
            away_mask &= df["home_team"].isin(opponents)

        home_tries  = df.loc[home_mask, "home_tries"] + df.loc[home_mask, "home_penalty_tries"]
        away_tries  = df.loc[away_mask, "away_tries"] + df.loc[away_mask, "away_penalty_tries"]
        home_points = df.loc[home_mask, "home_score"]
        away_points = df.loc[away_mask, "away_score"]
        home_pens   = df.loc[home_mask, "home_penalties"]
        away_pens   = df.loc[away_mask, "away_penalties"]

        #combine home and away figures into a single series per stat
        tries  = pd.concat([home_tries, away_tries])
        points = pd.concat([home_points, away_points])
        pens   = pd.concat([home_pens, away_pens])

        return {
            "tries":  (tries.mean(), tries.std(), len(tries)),
            "points": (points.mean(), points.std(), len(points)),
            "pens":   (pens.mean(), pens.std(), len(pens)),
        }

    def plot_posterior_stats(self, teams=None, opponents=None, ncols=3, figsize_per_plot=(6, 4),
                              show_observed=True):
        """
        Plot posterior predictive distributions of tries, points, and penalties
        for each focal team, pooling samples across matchups against a set of
        opponents (both home and away). Optionally overlays observed mean (solid
        line) and +/-1 SD (dashed lines) from the training data, per team.
        """
        #plot focal teams if not supplied
        teams_to_plot = teams or sorted(self.focalTeams, key=lambda t: self.team_to_idx[t])
        opponent_pool = opponents or teams_to_plot


        stats = {"tries": {}, "points": {}, "pens": {}}
        observed = {"tries": {}, "points": {}, "pens": {}}


        for team in teams_to_plot: # loop teams
            t_idx = self.team_to_idx[team]

            tries_samples = []
            points_samples = []
            pens_samples = []

            for opp in opponent_pool: # loop opponents
                if opp == team: # skip
                    continue
                o_idx = self.team_to_idx[opp]

                #home
                sh, sa, th, ta, ph, pa ,*_ = self.predict_match(t_idx, o_idx) # NOTE wont work for KISS models 

                tries_samples.append(th)
                points_samples.append(sh)
                pens_samples.append(ph)

                #away
                sh2, sa2, th2, ta2, ph2, pa2,*_= self.predict_match(o_idx, t_idx)

                tries_samples.append(ta2)
                points_samples.append(sa2)
                pens_samples.append(pa2)

            if len(tries_samples) == 0:
                print(f"Skipping {team}: no valid opponents in opponent pool.")
                continue

            #pool all home+away samples for this team into one array per component
            stats["tries"][team]  = np.concatenate(tries_samples)
            stats["points"][team] = np.concatenate(points_samples)
            stats["pens"][team]   = np.concatenate(pens_samples)

            if show_observed: #get observed data 
                obs = self._observed_team_stats(team, opponents=opponent_pool if opponents else None)
                for component in ["tries", "points", "pens"]:
                    observed[component][team] = obs[component]

        plotted_teams = list(stats["tries"].keys())
        component_names = list(stats.keys())
        ncols = min(ncols, len(component_names))
        nrows = int(np.ceil(len(component_names) / ncols))


        #plot
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
            squeeze=False
        )
        axes = axes.flatten()

        #colour map
        colors = plt.get_cmap(TEAM_CMAP)(np.linspace(0, 1, len(plotted_teams)))

        for ax_idx, component in enumerate(component_names):
            ax = axes[ax_idx]
            for team, color in zip(plotted_teams, colors):
                sns.histplot(stats[component][team], ax=ax, label=team, color=color, linewidth=1.5, alpha=0.5)

                #overlay observed mean and +/-1 SD as reference lines
                if show_observed and team in observed[component]:
                    mean, sd, n = observed[component][team]
                    if not np.isnan(mean):
                        ax.axvline(mean, color=color, linestyle="-", linewidth=1, alpha=0.8)
                        if not np.isnan(sd):
                            ax.axvline(mean - sd, color=color, linestyle="--", linewidth=0.8, alpha=0.5)
                            ax.axvline(mean + sd, color=color, linestyle="--", linewidth=0.8, alpha=0.5)

            ax.set_title(component.capitalize())
            ax.set_xlabel(component.capitalize())
            ax.set_ylabel("Density")

        for ax_idx in range(len(component_names), len(axes)):
            axes[ax_idx].axis("off")

        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes[:len(component_names)]:
            leg = ax.get_legend()
            if leg:
                leg.remove()

        #build a combined legend including the observed-line explainer entries
        legend_elements = handles.copy()
        legend_labels = labels.copy()
        if show_observed:
            from matplotlib.lines import Line2D
            legend_elements += [
                Line2D([0], [0], color=OBSERVED_COLOR, linestyle="-", linewidth=1),
                Line2D([0], [0], color=OBSERVED_COLOR, linestyle="--", linewidth=0.8),
            ]
            legend_labels += ["Observed mean", "Observed +/-1 SD"]

        fig.legend(legend_elements, legend_labels, loc="center left", bbox_to_anchor=(1.0, 0.5), title="Team")

        plt.tight_layout()
        plt.show()

        return stats, observed
            
    def simulate_group(self, verbose = True):
        """
        Runs full posterior predictive tournament simulation for initial stage of nations championship 2026.

        Returns
        -------
        northDf, southDf
            DataFrames of competition points
            (shape: nSamples x nTeams)
        """
        
        fixtures = [
            ("All Blacks","France"),
            ("Japan","France"),
            ("Australia","Ireland"),
            ("Wales","Fiji"),
            ("South Africa","England"),
            ("Argentina","Scotland"),

            ("All Blacks","Italy"),
            ("Australia","France"),
            ("Japan","Ireland"),
            ("England","Fiji"),
            ("South Africa","Scotland"),
            ("Argentina","Wales"),

            ("All Blacks","Ireland"),
            ("Japan","France"),
            ("Australia","Italy"),
            ("Scotland","Fiji"),
            ("South Africa","Wales"),
            ("Argentina","England"),

            ("Ireland","Argentina"),
            ("Italy","South Africa"),
            ("Scotland","All Blacks"),
            ("Wales","Japan"),
            ("France","Fiji"),
            ("England","Australia"),

            ("France","South Africa"),
            ("Italy","Argentina"),
            ("Wales","All Blacks"),
            ("England","Japan"),
            ("Ireland","Fiji"),
            ("Scotland","Australia"),

            ("England","All Blacks"),
            ("Scotland","Japan"),
            ("Ireland","South Africa"),
            ("Italy","Fiji"),
            ("France","Argentina"),
            ("Wales","Australia")
        ]

        northTeams = ["England", "Ireland", "France", "Scotland", "Wales", "Italy"]
        southTeams = ["All Blacks", "South Africa", "Australia", "Argentina", "Japan", "Fiji"]

        allTeams = list(self.team_to_idx.keys())
        nSamples = self.att.shape[0]

        #initalise points and points difference
        points = {t: np.zeros(nSamples, dtype=int) for t in allTeams}
        pointsDiff = {t: np.zeros(nSamples, dtype=int) for t in allTeams}
        
        #loop fixtures
        for homeTeam, awayTeam in fixtures:
            #get team indexes
            hi = self.team_to_idx[homeTeam]
            ai = self.team_to_idx[awayTeam]

            #only need score and n tries
            scoreHome, scoreAway, triesHome, triesAway, *_ = self.predict_match(hi, ai)

            #get table points from score and tries
            ptsHome, ptsAway = self._match_points(
                scoreHome, scoreAway, triesHome, triesAway
            )

            #add points to tournament score for each team
            points[homeTeam] += ptsHome
            points[awayTeam] += ptsAway

            # points difference
            pointsDiff[homeTeam] += scoreHome - scoreAway
            pointsDiff[awayTeam] += scoreAway - scoreHome


        northDf = pd.DataFrame({t: points[t] for t in northTeams})
        southDf = pd.DataFrame({t: points[t] for t in southTeams})

        northPdDf = pd.DataFrame({t: pointsDiff[t] for t in northTeams})
        southPdDf = pd.DataFrame({t: pointsDiff[t] for t in southTeams})
        if verbose:
            print("=== Northern Hemisphere ===")
            print(self.table_summary(northDf,northPdDf).round(3))

            print("\n=== Southern Hemisphere ===")
            print(self.table_summary(southDf,southPdDf).round(3))


            #helper for labels
            def _ordinal(n):
                if 11 <= n % 100 <= 13:
                    return f"{n}th"
                return f"{n}{ {1:'st',2:'nd',3:'rd'}.get(n % 10,'th') }"


            #plots probs of each postion for each team
            def _plot_positions(pointsDf, pdDf, title):

                ranks = self._get_ranks(pointsDf, pdDf)

                nTeams = pointsDf.shape[1]
                order = pointsDf.mean().sort_values(ascending=False).index.tolist()

                #probability of finishing in each position, per team
                probs = {}
                for team in order:
                    counts = ranks[team].value_counts(normalize=True)
                    probs[team] = np.array(
                        [counts.get(r, 0.0) for r in range(1, nTeams + 1)]
                    )

                #diverging colour scale: green = good position, red = bad position
                colors = plt.get_cmap(POSITION_CMAP)(np.linspace(0, 1, nTeams))

                fig, ax = plt.subplots(figsize=(9,5) )

                for i, team in enumerate(order):

                    left = 0

                    for r in range(nTeams):

                        p = probs[team][r]

                        ax.barh(i, p, left=left, color=colors[r], height=0.65, edgecolor="white", linewidth=0.5)

                        if p > 0.06:
                            ax.text(
                                left + p / 2,
                                i,
                                f"{p:.0%}",
                                ha="center",
                                va="center",
                                color="white",
                                fontsize=8,
                                fontweight="bold",
                            )

                        left += p

                ax.set_yticks(np.arange(len(order)))
                ax.set_yticklabels(order)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                ax.set_title(title)
                ax.invert_yaxis()


                legend = [
                    Patch(facecolor=colors[r], label=_ordinal(r + 1))
                    for r in range(nTeams)
                ]

                ax.legend(
                    handles=legend,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.15),
                    ncol=min(nTeams, 6),
                )

                plt.tight_layout()
                plt.show()

            _plot_positions(
                northDf,
                northPdDf,
                "Northern Hemisphere — finishing position probabilities",
            )

            _plot_positions(
                southDf,
                southPdDf,
                "Southern Hemisphere — finishing position probabilities",
            )

        return northDf, southDf
                
                
    def _get_ranks(self,pointsDf, pdDf):
        """Rank by competition points then points difference."""
        rankDf = pd.DataFrame(index=pointsDf.index, columns=pointsDf.columns)

        #for each sample, rank teams by points then points difference
        for i in pointsDf.index:
            ordering = (
                pd.DataFrame({
                    "pts": pointsDf.loc[i],
                    "pd": pdDf.loc[i]
                })
                .sort_values(["pts", "pd"], ascending=False)
                .index
            )

            rankDf.loc[i, ordering] = np.arange(1, len(ordering) + 1)

        return rankDf.astype(int)
    
    def _match_points(self, scoreHome, scoreAway, triesHome, triesAway):
        #competition points arrays, one entry per posterior sample
        ptsH = np.zeros(len(scoreHome), dtype=int)
        ptsA = np.zeros(len(scoreAway), dtype=int)

        #who won each sampled match
        homeWin = scoreHome > scoreAway
        awayWin = scoreAway > scoreHome
        draw = scoreHome == scoreAway

        #4 points for a win
        ptsH[homeWin] += 4
        ptsA[awayWin] += 4
        #2 points each for a draw
        ptsH[draw] += 2
        ptsA[draw] += 2

        #bonus point for scoring 4+ tries
        ptsH[triesHome >= 4] += 1
        ptsA[triesAway >= 4] += 1

        #bonus point for losing by 7 or fewer
        ptsH[awayWin & ((scoreAway - scoreHome) <= 7)] += 1
        ptsA[homeWin & ((scoreHome - scoreAway) <= 7)] += 1

        return ptsH, ptsA
    
    def table_summary(self, ptsDf, pdDf):

        ranks = self._get_ranks(ptsDf, pdDf)

        #summary of mean points and probability of finishing in each top-N bracket
        return pd.DataFrame({
            "mean_pts": ptsDf.mean(),
            "p_1st":  (ranks == 1).mean(),
            "p_top2": (ranks <= 2).mean(),
            "p_top3": (ranks <= 3).mean(),
            "p_top4": (ranks <= 4).mean(),
            "p_top5": (ranks <= 5).mean(),
        }).sort_values("mean_pts", ascending=False)
    
    def plot_pairwise_win_probs(self, teams=None, neutral=False):
        """
        Computes and plots home/away win probabilities for every team pair.

        Returns:
            winProbDf: DataFrame (home team x away team)
        """

        import pandas as pd
        import numpy as np

        #default to focal teams, sorted alphabetically
        teams = sorted(teams) if teams is not None else sorted(self.focalTeams)
        nTeams = len(teams)

        winMat = np.zeros((nTeams, nTeams), dtype=float)

        #loop every home/away team combination
        for i, homeTeam in enumerate(teams):
            for j, awayTeam in enumerate(teams):

                #no self-fixtures
                if homeTeam == awayTeam:
                    winMat[i, j] = np.nan
                    continue

                hi = self.team_to_idx[homeTeam]
                ai = self.team_to_idx[awayTeam]

                sh, sa, *_ = self.predict_match(hi, ai)#, neutral=neutral)

                winMat[i, j] = np.mean(sh > sa)

        winProbDf = pd.DataFrame(winMat, index=teams, columns=teams)

        # ---- plot ----
        plt.figure(figsize=(10, 8))
        sns.heatmap(winProbDf, annot=True, fmt=".2f",cmap=DIVERGING_CMAP, vmin=0, vmax=1, square=True,
                    linewidths=0.5, linecolor="white", cbar_kws={"label": "P(home win)"})
        plt.title("Home win probability (row beats column)")
        plt.xlabel("Away team")
        plt.ylabel("Home team")
        plt.tight_layout()
        plt.show()

        return winProbDf
    
    def plot_fixture(self, home_team, away_team, actual=None, ncols=2, figsize_per_plot=(6, 4)):
        """
        Plots posterior predictive distributions (tries, penalties, conversions,
        score) for both teams in a single fixture, overlaid per component.

        Parameters
        ----------
        home_team, away_team : str
        actual : dict-like, optional
            A row (dict, pd.Series, DataFrame row) with observed columns
            (home_score, away_score, home_tries, home_penalty_tries, etc.)
            to overlay as dashed vertical lines. Omit for fixtures with no result yet.
        """
        hi = self.team_to_idx[home_team]
        ai = self.team_to_idx[away_team]

        result = self.predict_match(hi, ai)
        sh, sa, th, ta, ph, pa, *rest = result
        ch, ca = (rest[-2], rest[-1]) if len(rest) >= 2 else (None, None)

        components = {
            "Score":     (sh, sa),
            "Tries":     (th, ta),
            "Penalties": (ph, pa),
        }
        if ch is not None:
            components["Conversions"] = (ch, ca)

        #helper to pull the matching observed value for a given component
        def get_actual(name):
            if actual is None:
                return None, None
            try:
                if name == "Tries":
                    ah = actual["home_tries"] + actual["home_penalty_tries"]
                    aa = actual["away_tries"] + actual["away_penalty_tries"]
                elif name == "Penalties":
                    ah, aa = actual["home_penalties"], actual["away_penalties"]
                elif name == "Conversions":
                    ah, aa = actual["home_conversions"], actual["away_conversions"]
                elif name == "Score":
                    ah, aa = actual["home_score"], actual["away_score"]
                else:
                    return None, None
                return ah, aa
            except (KeyError, TypeError):
                return None, None

        n = len(components)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
                                squeeze=False)
        axes = axes.flatten()

        for ax_idx, (name, (home_vals, away_vals)) in enumerate(components.items()):
            ax = axes[ax_idx]
            home_vals = np.asarray(home_vals)
            away_vals = np.asarray(away_vals)

            sns.histplot(home_vals, ax=ax, color=HOME_COLOR, label=home_team,
                        stat="density", discrete=True, alpha=0.55, edgecolor="white")
            sns.histplot(away_vals, ax=ax, color=AWAY_COLOR, label=away_team,
                        stat="density", discrete=True, alpha=0.55, edgecolor="white")

            #overlay observed values as dashed lines, if provided
            actual_h, actual_a = get_actual(name)
            if actual_h is not None:
                ax.axvline(actual_h, color=HOME_COLOR, linestyle="--", linewidth=2)
            if actual_a is not None:
                ax.axvline(actual_a, color=AWAY_COLOR, linestyle="--", linewidth=2)

            ax.set_title(name)
            ax.set_xlabel(name)
            ax.set_ylabel("Density")
            ax.legend()

        for ax_idx in range(n, len(axes)):
            axes[ax_idx].axis("off")

        fig.suptitle(f"{home_team} vs {away_team}", fontsize=14)
        plt.tight_layout()
        plt.show()

        print(f"\n{home_team} vs {away_team}\n")
        for name, (home_vals, away_vals) in components.items():
            home_vals = np.asarray(home_vals)
            away_vals = np.asarray(away_vals)
            print(f"{name:12s}: {home_team} = {home_vals.mean():.2f} +/- {home_vals.std():.2f}"
                f"   |   {away_team} = {away_vals.mean():.2f} +/- {away_vals.std():.2f}")

        #outcome probabilities from the sampled scores
        home_win = np.mean(sh > sa)
        draw     = np.mean(sh == sa)
        away_win = np.mean(sa > sh)
        print(f"\nP({home_team} win) = {home_win:.3f}   "
            f"P(draw) = {draw:.3f}   P({away_team} win) = {away_win:.3f}")
        print(f"Predicted score: {home_team} {np.mean(sh):.1f} - {np.mean(sa):.1f} {away_team}")

        return {"sh": sh, "sa": sa, "th": th, "ta": ta, "ph": ph, "pa": pa, "ch": ch, "ca": ca}
        
    def simulate_finals(self, northDf, southDf):
        """
        Simulates the finals. PLots win proabbilities for each match up based on north and south rankings. PLots probabilities of each hemisphere winning gievn these rankings.
        """
        nSamples = self.att.shape[0]

        #rank each hemisphere by mean points to decide the finals pairings
        north_rank = northDf.mean().sort_values(ascending=False).index.tolist()
        south_rank = southDf.mean().sort_values(ascending=False).index.tolist()

        fixtures = list(zip(north_rank, south_rank))

        north_wins_total = np.zeros(nSamples, dtype=int)
        south_wins_total = np.zeros(nSamples, dtype=int)

        win_probs = {}   # label -> (P(north), P(draw), P(south))
        scores = {}      # label -> (north_score_samples, south_score_samples, north_team, south_team)

        for fixtureIdx, (north_team, south_team) in enumerate(fixtures):

            # First-place match worth 2 points, all others 1 point
            matchPoints = 2 if fixtureIdx == 0 else 1

            #England has home advantage, all other matches neutral
            if north_team == "England":
                home, away, neutral = north_team, south_team, False
            elif south_team == "England":
                home, away, neutral = south_team, north_team, False
            else:
                home, away, neutral = north_team, south_team, True

            hi = self.team_to_idx[home]
            ai = self.team_to_idx[away]

            sh, sa, *_ = self.predict_match(hi, ai, neutral=neutral)

            #re-map scores back onto north/south rather than home/away
            if home == north_team:
                north_score, south_score = sh, sa
            else:
                north_score, south_score = sa, sh

            north_win = north_score > south_score
            south_win = south_score > north_score
            draw = north_score == south_score

            north_wins_total += north_win.astype(int) * matchPoints
            south_wins_total += south_win.astype(int) * matchPoints

            label = f"{north_team} vs {south_team}"
            win_probs[label] = (
                float(np.mean(north_win)),
                float(np.mean(draw)),
                float(np.mean(south_win)),
            )
            scores[label] = (north_score, south_score, north_team, south_team)

        #overall result decided by which hemisphere wins more individual matches
        north_hemi_win = north_wins_total > south_wins_total
        south_hemi_win = south_wins_total > north_wins_total
        tie = north_wins_total == south_wins_total

        overall = {
            "P(North)": float(np.mean(north_hemi_win)),
            "P(Tie)":   float(np.mean(tie)),
            "P(South)": float(np.mean(south_hemi_win)),
        }

        # ---- helper: diverging win/draw/loss bars, draw centered ----
        def _wdl_bar_plot(labels, probs, left_name, right_name, title, figsize=None):
            n = len(labels)
            figsize = figsize or (9, max(2, 0.8 * n))
            fig, ax = plt.subplots(figsize=figsize)

            y = np.arange(n)
            for i, label in enumerate(labels):
                p_left, p_draw, p_right = probs[label]
                ax.barh(i, p_left, left=0, color=HOME_COLOR, height=0.6, edgecolor="white", linewidth=0.5)
                ax.barh(i, p_draw, left=p_left, color=DRAW_COLOR, height=0.6, edgecolor="white", linewidth=0.5)
                ax.barh(i, p_right, left=p_left + p_draw, color=AWAY_COLOR, height=0.6, edgecolor="white", linewidth=0.5)

                #label each segment with its probability if there's room
                if p_left > 0.06:
                    ax.text(p_left / 2, i, f"{p_left:.0%}", va="center", ha="center",
                            color="white", fontsize=9, fontweight="bold")
                if p_draw > 0.06:
                    ax.text(p_left + p_draw / 2, i, f"{p_draw:.0%}", va="center", ha="center",
                            color="white", fontsize=9, fontweight="bold")
                if p_right > 0.06:
                    ax.text(p_left + p_draw + p_right / 2, i, f"{p_right:.0%}", va="center", ha="center",
                            color="white", fontsize=9, fontweight="bold")

            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title(title)
            ax.invert_yaxis()

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=HOME_COLOR, label=left_name),
                Patch(facecolor=DRAW_COLOR, label="Draw"),
                Patch(facecolor=AWAY_COLOR, label=right_name),
            ]
            ax.legend(handles=legend_elements, loc="upper center",
                    bbox_to_anchor=(0.5, -0.15), ncol=3)

            plt.tight_layout()
            plt.show()

        # ---- plot 1: per-fixture win/draw/loss ----
        fixture_labels = list(win_probs.keys())
        _wdl_bar_plot(fixture_labels, win_probs, "NH team win", "SH team win",
                    "Finals fixture outcomes",figsize=(9,5))

        # ---- plot 2: overall hemisphere result ----
        overall_probs = {"NH vs SH": (overall["P(North)"], overall["P(Tie)"], overall["P(South)"])}
        _wdl_bar_plot(["NH vs SH"], overall_probs, "NH wins",
                    "SH wins", "Overall finals result", figsize=(9, 2.2))

        # ---- plot 3: score distributions per fixture ----
        n = len(fixture_labels)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
        axes = axes.flatten()

        for i, label in enumerate(fixture_labels):
            north_score, south_score, north_team, south_team = scores[label]
            ax = axes[i]
            sns.histplot(north_score, ax=ax, color=HOME_COLOR, label=north_team,
                        stat="density", discrete=True, alpha=0.55, edgecolor="white")
            sns.histplot(south_score, ax=ax, color=AWAY_COLOR, label=south_team,
                        stat="density", discrete=True, alpha=0.55, edgecolor="white")
            ax.set_title(label)
            ax.set_xlabel("Score")
            ax.set_ylabel("Density")
            ax.legend()

        for i in range(n, len(axes)):
            axes[i].axis("off")

        fig.suptitle("Predicted score distributions — finals fixtures", fontsize=14)
        plt.tight_layout()
        plt.show()

        print("=== Finals fixtures ===")
        for label in fixture_labels:
            p_n, p_d, p_s = win_probs[label]
            print(f"{label}: P(North)={p_n:.3f}  P(Draw)={p_d:.3f}  P(South)={p_s:.3f}")

        print(f"\nOverall: P(NH wins=) = {overall['P(North)']:.3f}   "
            f"P(Tie) = {overall['P(Tie)']:.3f}   P(SH wins=) = {overall['P(South)']:.3f}")

        return {"fixtures": fixtures, "win_probs": win_probs, "overall": overall, "scores": scores}

    def simulate_tournament(self):
        #run the group stage then feed the results straight into the finals
        north,south = self.simulate_group()
        self.simulate_finals(north,south)


    def simulate_finals_from_scratch(self):
        """
        Fully sample-consistent finals simulation: for each posterior draw,
        determines who *actually* finished in each group position in that draw
        (not just by mean), then simulates the finals matchup for that draw's
        actual pairing. Neutral venue except England, who gets home advantage
        whenever they're in the fixture.

        Parameters
        ----------
        northDf, southDf : pd.DataFrame
            Output of simulate_group (nSamples x nTeams points).
        n_teams : int
            Number of ranked finals slots per hemisphere (e.g. 4 = top 4 vs top 4).

        Returns
        -------
        dict with 'win_probs' (per rank-slot), 'overall', 'scores' (per rank-slot arrays)
        """
        northDf,southDf = self.simulate_group(verbose=False)
        nSamples = self.att.shape[0]
        north_teams_all = list(northDf.columns)
        south_teams_all = list(southDf.columns)

        # unique rank per sample (ties broken by fixed column order)
        north_ranks = northDf.rank(axis=1, method="first", ascending=False).astype(int)
        south_ranks = southDf.rank(axis=1, method="first", ascending=False).astype(int)

        # for each rank r, which team holds that rank in each sample
        def _team_for_rank(ranks_df, r):
            return ranks_df.eq(r).idxmax(axis=1).values  # array of team names, len nSamples

        north_rank_team = {r: _team_for_rank(north_ranks, r) for r in range(1, 7)}
        south_rank_team = {r: _team_for_rank(south_ranks, r) for r in range(1, 7)}

        # precompute scores for every possible north-vs-south pair (36 calls for 6x6)
        pair_scores = {}  # (north_team, south_team) -> (north_score_arr, south_score_arr)
        for nt in north_teams_all:
            for st in south_teams_all:
                #England always gets home advantage when involved, else neutral venue
                if nt == "England":
                    home, away, neutral = nt, st, False
                elif st == "England":
                    home, away, neutral = st, nt, False
                else:
                    home, away, neutral = nt, st, True

                hi = self.team_to_idx[home]
                ai = self.team_to_idx[away]
                sh, sa, *_ = self.predict_match(hi, ai, neutral=neutral)

                #re-map scores back onto north/south rather than home/away
                if home == nt:
                    pair_scores[(nt, st)] = (sh, sa)
                else:
                    pair_scores[(nt, st)] = (sa, sh)

        # for each rank slot, gather per-sample scores based on that sample's actual pairing
        win_probs = {}
        scores = {}
        north_wins_total = np.zeros(nSamples, dtype=int)
        south_wins_total = np.zeros(nSamples, dtype=int)

        for r in range(1, 7):
            # 1st-place playoff worth double points
            matchPoints = 2 if r == 1 else 1
            nt_arr = north_rank_team[r]
            st_arr = south_rank_team[r]

            north_score = np.empty(nSamples)
            south_score = np.empty(nSamples)

            #for every possible pairing at this rank, fill in the samples that match it
            for nt in north_teams_all:
                nt_mask = (nt_arr == nt)
                if not nt_mask.any():
                    continue
                for st in south_teams_all:
                    mask = nt_mask & (st_arr == st)
                    if not mask.any():
                        continue
                    ns, ss = pair_scores[(nt, st)]
                    north_score[mask] = ns[mask]
                    south_score[mask] = ss[mask]

            north_win = north_score > south_score
            south_win = south_score > north_score
            draw = north_score == south_score

            north_wins_total += north_win.astype(int) * matchPoints
            south_wins_total += south_win.astype(int) * matchPoints

            label = f"North rank {r} vs South rank {r}"
            win_probs[label] = (
                float(np.mean(north_win)),
                float(np.mean(draw)),
                float(np.mean(south_win)),
            )
            scores[label] = (north_score, south_score)

        #overall result decided by which hemisphere wins more individual matches
        north_hemi_win = north_wins_total > south_wins_total
        south_hemi_win = south_wins_total > north_wins_total
        tie = north_wins_total == south_wins_total

        overall = {
            "P(North)": float(np.mean(north_hemi_win)),
            "P(Tie)":   float(np.mean(tie)),
            "P(South)": float(np.mean(south_hemi_win)),
        }

        # ---- reuse the same diverging bar plot style ----
        def _wdl_bar_plot(labels, probs, left_name, right_name, title, figsize=None):
            n = len(labels)
            figsize = figsize or (9, max(2, 0.8 * n))
            fig, ax = plt.subplots(figsize=figsize)
            y = np.arange(n)
            for i, label in enumerate(labels):
                p_left, p_draw, p_right = probs[label]
                ax.barh(i, p_left, left=0, color=HOME_COLOR, height=0.6, edgecolor="white", linewidth=0.5)
                ax.barh(i, p_draw, left=p_left, color=DRAW_COLOR, height=0.6, edgecolor="white", linewidth=0.5)
                ax.barh(i, p_right, left=p_left + p_draw, color=AWAY_COLOR, height=0.6, edgecolor="white", linewidth=0.5)
                for val, off in [(p_left, p_left / 2), (p_draw, p_left + p_draw / 2),
                                (p_right, p_left + p_draw + p_right / 2)]:
                    if val > 0.06:
                        ax.text(off, i, f"{val:.0%}", va="center", ha="center",
                                color="white", fontsize=9, fontweight="bold")
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability")
            ax.set_title(title)
            ax.invert_yaxis()
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=HOME_COLOR, label=left_name),
                                Patch(facecolor=DRAW_COLOR, label="Draw"),
                                Patch(facecolor=AWAY_COLOR, label=right_name)]
            ax.legend(handles=legend_elements, loc="upper center",
                    bbox_to_anchor=(0.5, -0.15), ncol=3)
            plt.tight_layout()
            plt.show()

            """        labels = list(win_probs.keys())
        _wdl_bar_plot(labels, win_probs, "North win", "South win",
                    "Finals outcomes by rank-slot (sample-consistent)")"""
        _wdl_bar_plot(["North vs South"],
                    {"North vs South": (overall["P(North)"], overall["P(Tie)"], overall["P(South)"])},
                    "NH wins", "SH wins",
                    "Overall finals result", figsize=(9, 2.2))

        print("=== Finals  ===")
        #for label in labels:
        #    p_n, p_d, p_s = win_probs[label]
        #    print(f"{label}: P(North)={p_n:.3f}  P(Draw)={p_d:.3f}  P(South)={p_s:.3f}")
        print(f"\nOverall: P(North) = {overall['P(North)']:.3f}   "
            f"P(Tie) = {overall['P(Tie)']:.3f}   P(South) = {overall['P(South)']:.3f}")

        return {"win_probs": win_probs, "overall": overall, "scores": scores}
    
    def plot_model_structure(self, **kwargs):

        if not hasattr(self, "model"):
            print("No fitted model found - call .fit(df) first.")
            return None
        try:
            graph = pm.model_to_graphviz(self.model, **kwargs)
        except ImportError:
            print("graphviz isn't installed - run `pip install graphviz` "
                "(and make sure the graphviz system package is on your PATH) to use this.")
            return None
        return graph

###################################################################################################################
#                                     DIFFERENT MODEL TYPES                                                       #
###################################################################################################################
class PoissonModel(RugbyModel):
    def __init__(self, sd_scale = 1 ,scale_int = 0.5, **kwargs):
        super().__init__(**kwargs)    # run RugbyModel.__init__ first
        self.sd_scale = sd_scale
        self.scale_int = scale_int

    def _build(self, temporal_weights, data): # need a function to unpack data - can this be in rugyb model class?
        home_idx,away_idx,obs_tries_home,obs_tries_away,obs_pens_home,obs_pens_away,obs_conv_home,obs_conv_away,obs_points_home,obs_points_away ,obs_dgs_home,obs_dgs_away= super()._unpack_data(data)
        muTry = np.log(np.mean(np.concatenate([obs_tries_away, obs_tries_home])))
        muPen = np.log(np.mean(np.concatenate([obs_pens_away, obs_pens_home])))
        muConv = (obs_conv_home.sum() + obs_conv_away.sum()) / (obs_tries_home.sum() + obs_tries_away.sum())
        missing_home = data.loc[~data["home_team"].isin(self.team_to_idx), "home_team"].unique()
        missing_away = data.loc[~data["away_team"].isin(self.team_to_idx), "away_team"].unique()
        if len(missing_home) or len(missing_away):
            raise ValueError(f"Teams not in index — home: {missing_home}, away: {missing_away}")
        with pm.Model() as model:
            #home advantage - one for all teams
            home = pm.Normal("home",mu = 0, sigma = 0.3) 
            
            #Tries----

            sd_att=pm.HalfNormal("sd_att", sigma=self.sd_scale)#half normal
            sd_def=pm.HalfNormal("sd_def", sigma=self.sd_scale)#ha;f normal

            att = pm.ZeroSumNormal("att", sigma=sd_att, shape=self.n_teams)
            def_ = pm.ZeroSumNormal("def_", sigma=sd_def, shape=self.n_teams)

            int_try =  pm.Normal("int_try", mu=muTry, sigma=self.scale_int) #normal

            lambda_home_tries =  pm.math.exp(int_try+att[home_idx]-def_[away_idx]+home) 
            lambda_away_tries = pm.math.exp(int_try+att[away_idx]-def_[home_idx]) 

            ll_home_try = pm.logp(pm.Poisson.dist(mu = lambda_home_tries), obs_tries_home) 
            ll_away_try = pm.logp(pm.Poisson.dist(mu = lambda_away_tries), obs_tries_away)

            pm.Potential("home_tries",   (temporal_weights * ll_home_try).sum())
            pm.Potential("away_tries", (temporal_weights * ll_away_try).sum())


            #pens----   

            sd_dis_clean = pm.HalfNormal("sd_dis_clean", sigma=self.sd_scale)#half normal
            sd_dis_force = pm.HalfNormal("sd_dis_force", sigma=self.sd_scale)#half normal


            dis_clean = pm.ZeroSumNormal("dis_clean",  sigma=sd_dis_clean, shape=self.n_teams)
            dis_force = pm.ZeroSumNormal("dis_force",  sigma=sd_dis_force, shape=self.n_teams)


            int_pen =  pm.Normal("int_pen", mu=muPen, sigma=self.scale_int) #normal

            lambda_home_pens =  pm.math.exp(int_pen+dis_force[home_idx]-dis_clean[away_idx]+home) 
            lambda_away_pens = pm.math.exp(int_pen+dis_force[away_idx]-dis_clean[home_idx]) 


            ll_home_pen= pm.logp(pm.Poisson.dist(mu = lambda_home_pens), obs_pens_home) 

            ll_away_pen = pm.logp(pm.Poisson.dist(mu = lambda_away_pens), obs_pens_away)

            pm.Potential("home_pens",   (temporal_weights * ll_home_pen).sum())
            pm.Potential("away_pens", (temporal_weights * ll_away_pen).sum())

            #conversions---

            beta_conv = pm.Beta("p_conv", mu=muConv, sigma=0.08,shape  = self.n_teams) # beta
            logp_conv_home = pm.logp(pm.Binomial.dist(n=obs_tries_home, p=beta_conv[home_idx]), obs_conv_home)
            logp_conv_away = pm.logp(pm.Binomial.dist(n=obs_tries_away, p=beta_conv[away_idx]), obs_conv_away)
            pm.Potential("home_conversions", (temporal_weights * logp_conv_home).sum()) # binomaial 
            pm.Potential("away_conversions", (temporal_weights * logp_conv_away).sum()) # binomaial
            self.model = model
            self.trace = pm.sample(self.nSamples//self.cores, tune=self.tune, cores=self.cores, target_accept = self.target_accept)
        self.unpack_posterior()
        
        
    def unpack_posterior(self):
        trace = self.trace
        
        n_samples = trace.posterior.dims["draw"] * trace.posterior.dims["chain"]

        def f(arr):
            return arr.values.reshape(n_samples, *arr.shape[2:])

        self.att       = f(trace.posterior["att"])
        self.def_      = f(trace.posterior["def_"])
        self.dis_clean = f(trace.posterior["dis_clean"])
        self.dis_force = f(trace.posterior["dis_force"])
        self.p_conv    = f(trace.posterior["p_conv"])
        self.int_try   = f(trace.posterior["int_try"])
        self.int_pen   = f(trace.posterior["int_pen"])
        self.home      = f(trace.posterior["home"])


    def predict_match(self, hi, ai,neutral = False):
        home = self.home if not neutral else 0
        lam_h = np.exp(self.int_try + self.att[:, hi] - self.def_[:, ai] + home)
        lam_a = np.exp(self.int_try + self.att[:, ai] - self.def_[:, hi])
        lam_h_pen = np.exp(self.int_pen + self.dis_force[:, hi] - self.dis_clean[:, ai] + home)
        lam_a_pen = np.exp(self.int_pen + self.dis_force[:, ai] - self.dis_clean[:, hi])
        th = np.random.poisson(lam_h);  ta = np.random.poisson(lam_a)
        ph = np.random.poisson(lam_h_pen); pa = np.random.poisson(lam_a_pen)
        ch = np.random.binomial(th, self.p_conv[:, hi])
        ca = np.random.binomial(ta, self.p_conv[:, ai])
        sh = th * 5 + ch * 2 + ph * 3
        sa = ta * 5 + ca * 2 + pa * 3
        return sh, sa, th, ta, ph, pa,ch,ca
    
    def plot_params(self):
        self.plot_team_params(params = ["att","def_","dis_clean","dis_force","p_conv","int_try","int_pen"])
    
    
class NegBinModel(RugbyModel):
    def __init__(self, sd_scale = 1,scale_int = 0.5 , **kwargs):
        super().__init__(**kwargs)    # run RugbyModel.__init__ first
        self.sd_scale = sd_scale
        self.scale_int = scale_int


    def _build(self, temporal_weights, data): # need a function to unpack data - can this be in rugyb model class?
        home_idx,away_idx,obs_tries_home,obs_tries_away,obs_pens_home,obs_pens_away,obs_conv_home,obs_conv_away,obs_points_home,obs_points_away,obs_dgs_home,obs_dgs_away = super()._unpack_data(data)
        
        muTry = np.log(np.mean(np.concatenate([obs_tries_away, obs_tries_home])))
        muPen = np.log(np.mean(np.concatenate([obs_pens_away, obs_pens_home])))
        muConv = (obs_conv_home.sum() + obs_conv_away.sum()) / (obs_tries_home.sum() + obs_tries_away.sum())
        missing_home = data.loc[~data["home_team"].isin(self.team_to_idx), "home_team"].unique()
        missing_away = data.loc[~data["away_team"].isin(self.team_to_idx), "away_team"].unique()
        if len(missing_home) or len(missing_away):
            raise ValueError(f"Teams not in index — home: {missing_home}, away: {missing_away}")
        with pm.Model() as model:
            #home advantage - one for all teams
            home = pm.Normal("home",mu = 0, sigma = 0.3) 
            
            sd_att=pm.HalfNormal("sd_att", sigma=self.sd_scale)#half normal
            sd_def=pm.HalfNormal("sd_def", sigma=self.sd_scale)#ha;f normal

            att = pm.ZeroSumNormal("att", sigma=sd_att, shape=self.n_teams)
            def_ = pm.ZeroSumNormal("def_", sigma=sd_def, shape=self.n_teams)

            
            int_try =  pm.Normal("int_try", mu=muTry, sigma=self.scale_int) #normal

            lambda_home_tries =  pm.math.exp(int_try+att[home_idx]-def_[away_idx]+home) 
            lambda_away_tries = pm.math.exp(int_try+att[away_idx]-def_[home_idx]) 

            alpha =  pm.Exponential("alpha",lam = 1.0)#overdispersion of negatice binomial 

            ll_home_try = pm.logp(pm.NegativeBinomial.dist(mu = lambda_home_tries, alpha = alpha), obs_tries_home) 

            ll_away_try = pm.logp(pm.NegativeBinomial.dist(mu = lambda_away_tries, alpha = alpha), obs_tries_away)
            pm.Potential("home_tries",   (temporal_weights * ll_home_try).sum())
            pm.Potential("away_tries", (temporal_weights * ll_away_try).sum())


            #pens----

            sd_dis_clean = pm.HalfNormal("sd_dis_clean", sigma=self.sd_scale)#half normal
            sd_dis_force = pm.HalfNormal("sd_dis_force", sigma=self.sd_scale)#half normal

            alphaPen = pm.Exponential("alphaPen",lam = 1.0)#overdispersion of negatice binomial 


            dis_clean = pm.ZeroSumNormal("dis_clean",  sigma=sd_dis_clean, shape=self.n_teams)
            dis_force = pm.ZeroSumNormal("dis_force", sigma=sd_dis_force, shape=self.n_teams)


            int_pen =  pm.Normal("int_pen", mu=muPen, sigma=self.scale_int) #normal

            lambda_home_pens =  pm.math.exp(int_pen+dis_force[home_idx]-dis_clean[away_idx]+home) 
            lambda_away_pens = pm.math.exp(int_pen+dis_force[away_idx]-dis_clean[home_idx]) 


            ll_home_pen= pm.logp(pm.NegativeBinomial.dist(mu = lambda_home_pens,alpha = alphaPen), obs_pens_home) 

            ll_away_pen = pm.logp(pm.NegativeBinomial.dist(mu = lambda_away_pens,alpha = alphaPen), obs_pens_away)

            pm.Potential("home_pens",   (temporal_weights * ll_home_pen).sum())
            pm.Potential("away_pens", (temporal_weights * ll_away_pen).sum())

            #conversions---

            beta_conv = pm.Beta("p_conv", mu=muConv, sigma=0.08,shape  = self.n_teams) # beta
            logp_conv_home = pm.logp(pm.Binomial.dist(n=obs_tries_home, p=beta_conv[home_idx]), obs_conv_home)
            logp_conv_away = pm.logp(pm.Binomial.dist(n=obs_tries_away, p=beta_conv[away_idx]), obs_conv_away)
            pm.Potential("home_conversions", (temporal_weights * logp_conv_home).sum()) # binomaial 
            pm.Potential("away_conversions", (temporal_weights * logp_conv_away).sum()) # binomaial
            self.model = model
            self.trace = pm.sample(self.nSamples//self.cores, tune=self.tune, cores=self.cores, target_accept = self.target_accept)
        self.unpack_posterior()
        
        
    def unpack_posterior(self):
        trace = self.trace
        
        n_samples = trace.posterior.dims["draw"] * trace.posterior.dims["chain"]

        def f(arr):
            return arr.values.reshape(n_samples, *arr.shape[2:])

        self.att       = f(trace.posterior["att"])
        self.def_      = f(trace.posterior["def_"])
        self.dis_clean = f(trace.posterior["dis_clean"])
        self.dis_force = f(trace.posterior["dis_force"])
        self.p_conv    = f(trace.posterior["p_conv"])
        self.int_try   = f(trace.posterior["int_try"])
        self.int_pen   = f(trace.posterior["int_pen"])
        self.home      = f(trace.posterior["home"])
        self.alpha = f(trace.posterior["alpha"])
        self.alphaPen = f(trace.posterior["alphaPen"])

    def predict_match(self, hi, ai,neutral = False):
        home = self.home if not neutral else 0
        lam_h = np.exp(self.int_try + self.att[:, hi] - self.def_[:, ai] + home)
        lam_a = np.exp(self.int_try + self.att[:, ai] - self.def_[:, hi])
        lam_h_pen = np.exp(self.int_pen + self.dis_force[:, hi] - self.dis_clean[:, ai] +home)
        lam_a_pen = np.exp(self.int_pen + self.dis_force[:, ai] - self.dis_clean[:, hi])

        # NegativeBinomial(mu, alpha) -> numpy's negative_binomial uses (n, p) parameterisation
        # n = alpha, p = alpha / (alpha + mu)
        n_try = self.alpha
        p_h_try = self.alpha / (self.alpha + lam_h)
        p_a_try = self.alpha / (self.alpha + lam_a)

        n_pen = self.alphaPen
        p_h_pen = self.alphaPen / (self.alphaPen + lam_h_pen)
        p_a_pen = self.alphaPen / (self.alphaPen + lam_a_pen)

        th = np.random.negative_binomial(n_try, p_h_try)
        ta = np.random.negative_binomial(n_try, p_a_try)
        ph = np.random.negative_binomial(n_pen, p_h_pen)
        pa = np.random.negative_binomial(n_pen, p_a_pen)

        ch = np.random.binomial(th, self.p_conv[:, hi])
        ca = np.random.binomial(ta, self.p_conv[:, ai])

        sh = th * 5 + ch * 2 + ph * 3
        sa = ta * 5 + ca * 2 + pa * 3
        return sh, sa, th, ta, ph, pa
    
    def plot_params(self):
        self.plot_team_params(params = ["att","def_","dis_clean","dis_force","p_conv","int_try","int_pen"])
class KISSModel(RugbyModel):
    def __init__(self, sd_scale = 1 ,scale_int = 0.5, **kwargs):
        super().__init__(**kwargs)    # run RugbyModel.__init__ first
        self.sd_scale = sd_scale
        self.scale_int = scale_int


    def _build(self, temporal_weights, data): # need a function to unpack data - can this be in rugyb model class?
        home_idx,away_idx,obs_tries_home,obs_tries_away,obs_pens_home,obs_pens_away,obs_conv_home,obs_conv_away,obs_points_home,obs_points_away,obs_dgs_home,obs_dgs_away = super()._unpack_data(data)
        
        muScore = np.log(np.mean(np.concatenate([obs_points_away, obs_points_home])))
        missing_home = data.loc[~data["home_team"].isin(self.team_to_idx), "home_team"].unique()
        missing_away = data.loc[~data["away_team"].isin(self.team_to_idx), "away_team"].unique()
        if len(missing_home) or len(missing_away):
            raise ValueError(f"Teams not in index — home: {missing_home}, away: {missing_away}")
        with pm.Model() as model:
            #home advantage - one for all teams
            home = pm.Normal("home",mu = 0, sigma = 0.3) 
            
            sd_att=pm.HalfNormal("sd_att", sigma=self.sd_scale)#half normal
            sd_def=pm.HalfNormal("sd_def", sigma=self.sd_scale)#ha;f normal

            att = pm.ZeroSumNormal("att",  sigma=sd_att, shape=self.n_teams)
            def_ = pm.ZeroSumNormal("def_",  sigma=sd_def, shape=self.n_teams)

            
            int_try =  pm.Normal("int", mu=muScore, sigma=self.scale_int) #normal

            lambda_home =  pm.math.exp(int_try+att[home_idx]-def_[away_idx]+home) 
            lambda_away = pm.math.exp(int_try+att[away_idx]-def_[home_idx]) 

            alpha =  pm.Exponential("alpha",lam = 1.0)#overdispersion of negatice binomial 

            ll_home_try = pm.logp(pm.NegativeBinomial.dist(mu = lambda_home, alpha = alpha), obs_points_home) 

            ll_away_try = pm.logp(pm.NegativeBinomial.dist(mu = lambda_away, alpha = alpha), obs_points_away)
            pm.Potential("home_points",   (temporal_weights * ll_home_try).sum())
            pm.Potential("away_points", (temporal_weights * ll_away_try).sum())

            self.model = model
            self.trace = pm.sample(self.nSamples//self.cores, tune=self.tune, cores=self.cores, target_accept = self.target_accept)
        self.unpack_posterior()
        
        
    def unpack_posterior(self):
        trace = self.trace
        
        n_samples = trace.posterior.dims["draw"] * trace.posterior.dims["chain"]

        def f(arr):
            return arr.values.reshape(n_samples, *arr.shape[2:])

        self.att       = f(trace.posterior["att"])
        self.def_      = f(trace.posterior["def_"])
        self.int   = f(trace.posterior["int"])
        self.home      = f(trace.posterior["home"])
        
        self.alpha = f(trace.posterior["alpha"])

    def predict_match(self, hi, ai,neutral = False):
        home = self.home if not neutral else 0
        lam_h = np.exp(self.int + self.att[:, hi] - self.def_[:, ai] + home)
        lam_a = np.exp(self.int + self.att[:, ai] - self.def_[:, hi])

        # NegativeBinomial(mu, alpha) -> numpy's negative_binomial uses (n, p) parameterisation
        # n = alpha, p = alpha / (alpha + mu)
        n_try = self.alpha
        p_h_try = self.alpha / (self.alpha + lam_h)
        p_a_try = self.alpha / (self.alpha + lam_a)

        sh = np.random.negative_binomial(n_try, p_h_try)
        sa = np.random.negative_binomial(n_try, p_a_try)

        return sh, sa
    
    def plot_params(self):
        self.plot_team_params(params = ["att","def_"])
class KISSSModel(RugbyModel):
    def __init__(self, sd_scale = 1 ,scale_int = 0.5, **kwargs):
        super().__init__(**kwargs)    # run RugbyModel.__init__ first
        self.sd_scale = sd_scale
        self.scale_int = scale_int


    def _build(self, temporal_weights, data): # need a function to unpack data - can this be in rugyb model class?
        home_idx,away_idx,obs_tries_home,obs_tries_away,obs_pens_home,obs_pens_away,obs_conv_home,obs_conv_away,obs_points_home,obs_points_away ,obs_dgs_home,obs_dgs_away= super()._unpack_data(data)
        
        muScore = np.log(np.mean(np.concatenate([obs_points_away, obs_points_home])))
        missing_home = data.loc[~data["home_team"].isin(self.team_to_idx), "home_team"].unique()
        missing_away = data.loc[~data["away_team"].isin(self.team_to_idx), "away_team"].unique()
        if len(missing_home) or len(missing_away):
            raise ValueError(f"Teams not in index — home: {missing_home}, away: {missing_away}")
        with pm.Model() as model:

            sd_att=pm.HalfNormal("sd_strength", sigma=self.sd_scale)#half normal

            strength = pm.ZeroSumNormal("strength",sigma=sd_att, shape=self.n_teams)
            int_try =  pm.Normal("int", mu=muScore, sigma=0.5) #normal

            lambda_home =  pm.math.exp(int_try+strength[home_idx]-strength[away_idx]) 
            lambda_away = pm.math.exp(int_try+strength[away_idx]-strength[home_idx]) 

            alpha =  pm.Exponential("alpha",lam = 1.0)#overdispersion of negatice binomial 

            ll_home_try = pm.logp(pm.NegativeBinomial.dist(mu = lambda_home, alpha = alpha), obs_points_home) 

            ll_away_try = pm.logp(pm.NegativeBinomial.dist(mu = lambda_away, alpha = alpha), obs_points_away)


            pm.Potential("home_points",   (temporal_weights * ll_home_try).sum())
            pm.Potential("away_points", (temporal_weights * ll_away_try).sum())
            self.model = model
        
            self.trace = pm.sample(self.nSamples//self.cores, tune=self.tune, cores=self.cores, target_accept = self.target_accept)
        self.unpack_posterior()
        
        
    def unpack_posterior(self):
        trace = self.trace
        
        n_samples = trace.posterior.dims["draw"] * trace.posterior.dims["chain"]

        def f(arr):
            return arr.values.reshape(n_samples, *arr.shape[2:])

        self.strength       = f(trace.posterior["strength"])
        self.int   = f(trace.posterior["int"])
        
        self.alpha = f(trace.posterior["alpha"])

    def predict_match(self, hi, ai,neutral = False):
        lam_h = np.exp(self.int + self.strength[:, hi] - self.strength[:, ai])
        lam_a = np.exp(self.int + self.strength[:, ai] - self.strength[:, hi])

        # NegativeBinomial(mu, alpha) -> numpy's negative_binomial uses (n, p) parameterisation
        # n = alpha, p = alpha / (alpha + mu)
        n_try = self.alpha
        p_h_try = self.alpha / (self.alpha + lam_h)
        p_a_try = self.alpha / (self.alpha + lam_a)

        sh = np.random.negative_binomial(n_try, p_h_try)
        sa = np.random.negative_binomial(n_try, p_a_try)

        return sh, sa
    
    def plot_params(self):
        self.plot_team_params(params = ["strength"])
        

class PoissonModel_DG(RugbyModel):
    def __init__(self, sd_scale = 1 ,scale_int = 0.5, **kwargs):
        super().__init__(**kwargs)    # run RugbyModel.__init__ first
        self.sd_scale = sd_scale
        self.scale_int = scale_int


    def _build(self, temporal_weights, data): # need a function to unpack data - can this be in rugyb model class?
        home_idx,away_idx,obs_tries_home,obs_tries_away,obs_pens_home,obs_pens_away,obs_conv_home,obs_conv_away,obs_points_home,obs_points_away ,obs_dgs_home,obs_dgs_away= super()._unpack_data(data)
        muTry = np.log(np.mean(np.concatenate([obs_tries_away, obs_tries_home])))
        muPen = np.log(np.mean(np.concatenate([obs_pens_away, obs_pens_home])))
        muConv = (obs_conv_home.sum() + obs_conv_away.sum()) / (obs_tries_home.sum() + obs_tries_away.sum())
        all_dg = np.concatenate([obs_dgs_home, obs_dgs_away])
        nonzero_dg = all_dg[all_dg > 0]
        if len(nonzero_dg) > 0:
            muDG = np.log(np.mean(nonzero_dg))
        else:
            muDG = np.log(0.5)  # fallback if no DGs
        missing_home = data.loc[~data["home_team"].isin(self.team_to_idx), "home_team"].unique()
        missing_away = data.loc[~data["away_team"].isin(self.team_to_idx), "away_team"].unique()
        if len(missing_home) or len(missing_away):
            raise ValueError(f"Teams not in index — home: {missing_home}, away: {missing_away}")
        with pm.Model() as model:
            #home advantage - one for all teams
            home = pm.Normal("home",mu = 0, sigma = 0.3) 
            
            #Tries----

            sd_att=pm.HalfNormal("sd_att", sigma=self.sd_scale)#half normal
            sd_def=pm.HalfNormal("sd_def", sigma=self.sd_scale)#ha;f normal

            att = pm.ZeroSumNormal("att", sigma=sd_att, shape=self.n_teams)
            def_ = pm.ZeroSumNormal("def_", sigma=sd_def, shape=self.n_teams)

            int_try =  pm.Normal("int_try", mu=muTry, sigma=self.scale_int) #normal

            lambda_home_tries =  pm.math.exp(int_try+att[home_idx]-def_[away_idx]+home) 
            lambda_away_tries = pm.math.exp(int_try+att[away_idx]-def_[home_idx]) 

            ll_home_try = pm.logp(pm.Poisson.dist(mu = lambda_home_tries), obs_tries_home) 
            ll_away_try = pm.logp(pm.Poisson.dist(mu = lambda_away_tries), obs_tries_away)

            pm.Potential("home_tries",   (temporal_weights * ll_home_try).sum())
            pm.Potential("away_tries", (temporal_weights * ll_away_try).sum())


            #pens----   

            sd_dis_clean = pm.HalfNormal("sd_dis_clean", sigma=self.sd_scale)#half normal
            sd_dis_force = pm.HalfNormal("sd_dis_force", sigma=self.sd_scale)#half normal


            dis_clean = pm.ZeroSumNormal("dis_clean",  sigma=sd_dis_clean, shape=self.n_teams)
            dis_force = pm.ZeroSumNormal("dis_force",  sigma=sd_dis_force, shape=self.n_teams)


            int_pen =  pm.Normal("int_pen", mu=muPen, sigma=self.scale_int) #normal

            lambda_home_pens =  pm.math.exp(int_pen+dis_force[home_idx]-dis_clean[away_idx]+home) 
            lambda_away_pens = pm.math.exp(int_pen+dis_force[away_idx]-dis_clean[home_idx]) 


            ll_home_pen= pm.logp(pm.Poisson.dist(mu = lambda_home_pens), obs_pens_home) 

            ll_away_pen = pm.logp(pm.Poisson.dist(mu = lambda_away_pens), obs_pens_away)

            pm.Potential("home_pens",   (temporal_weights * ll_home_pen).sum())
            pm.Potential("away_pens", (temporal_weights * ll_away_pen).sum())

            #conversions---

            beta_conv = pm.Beta("p_conv", mu=muConv, sigma=0.08,shape  = self.n_teams) # beta
            logp_conv_home = pm.logp(pm.Binomial.dist(n=obs_tries_home, p=beta_conv[home_idx]), obs_conv_home)
            logp_conv_away = pm.logp(pm.Binomial.dist(n=obs_tries_away, p=beta_conv[away_idx]), obs_conv_away)
            pm.Potential("home_conversions", (temporal_weights * logp_conv_home).sum()) # binomaial 
            pm.Potential("away_conversions", (temporal_weights * logp_conv_away).sum()) # binomaial
            
            #DGs
            # Drop goals ----
            sd_att_dg = pm.HalfNormal("sd_att_dg", sigma=self.sd_scale)
            att_dg = pm.ZeroSumNormal("att_dg", sigma=sd_att_dg, shape=self.n_teams)

            int_dg = pm.Normal("int_dg", mu=muDG, sigma=0.5)  # muDG from log of nonzero rate, see below

            # psi: probability a team is even "in range" to attempt/succeed a drop goal
            # model on logit scale, can be per-team or global
            psi_logit = pm.Normal("psi_logit_dg", mu=0, sigma=1, shape=self.n_teams)
            psi = pm.math.sigmoid(psi_logit)

            lambda_home_dg = pm.math.exp(int_dg + att_dg[home_idx] + home)
            lambda_away_dg = pm.math.exp(int_dg + att_dg[away_idx])

            psi_home = psi[home_idx]
            psi_away = psi[away_idx]

            ll_home_dg = pm.logp(
                pm.ZeroInflatedPoisson.dist(
                    psi=psi_home,
                    mu=lambda_home_dg
                ),
                obs_dgs_home
            )

            ll_away_dg = pm.logp(
                pm.ZeroInflatedPoisson.dist(
                    psi=psi_away,
                    mu=lambda_away_dg
                ),
                obs_dgs_away
            )
            self.model = model
            self.trace = pm.sample(self.nSamples//self.cores, tune=self.tune, cores=self.cores, target_accept = self.target_accept)
        self.unpack_posterior()
        
    def unpack_posterior(self):
        trace = self.trace
        
        n_samples = trace.posterior.dims["draw"] * trace.posterior.dims["chain"]

        def f(arr):
            return arr.values.reshape(n_samples, *arr.shape[2:])

        self.att       = f(trace.posterior["att"])
        self.def_      = f(trace.posterior["def_"])
        self.dis_clean = f(trace.posterior["dis_clean"])
        self.dis_force = f(trace.posterior["dis_force"])
        self.p_conv    = f(trace.posterior["p_conv"])
        self.int_try   = f(trace.posterior["int_try"])
        self.int_pen   = f(trace.posterior["int_pen"])
        self.home      = f(trace.posterior["home"])

        # drop goals
        self.att_dg       = f(trace.posterior["att_dg"])
        self.int_dg        = f(trace.posterior["int_dg"])
        self.psi_logit_dg  = f(trace.posterior["psi_logit_dg"])

    def predict_match(self, hi, ai,neutral = False):
        home = self.home if not neutral else 0
        lam_h = np.exp(self.int_try + self.att[:, hi] - self.def_[:, ai] +home)
        lam_a = np.exp(self.int_try + self.att[:, ai] - self.def_[:, hi])
        lam_h_pen = np.exp(self.int_pen + self.dis_force[:, hi] - self.dis_clean[:, ai] + home)
        lam_a_pen = np.exp(self.int_pen + self.dis_force[:, ai] - self.dis_clean[:, hi])
        th = np.random.poisson(lam_h);  ta = np.random.poisson(lam_a)
        ph = np.random.poisson(lam_h_pen); pa = np.random.poisson(lam_a_pen)
        ch = np.random.binomial(th, self.p_conv[:, hi])
        ca = np.random.binomial(ta, self.p_conv[:, ai])

        # drop goals — zero-inflated Poisson
        lam_h_dg = np.exp(self.int_dg + self.att_dg[:, hi] + home)
        lam_a_dg = np.exp(self.int_dg + self.att_dg[:, ai])

        psi_h = 1 / (1 + np.exp(-self.psi_logit_dg[:, hi]))
        psi_a = 1 / (1 + np.exp(-self.psi_logit_dg[:, ai]))

        active_h = np.random.binomial(1, psi_h)
        active_a = np.random.binomial(1, psi_a)

        dh = active_h * np.random.poisson(lam_h_dg)
        da = active_a * np.random.poisson(lam_a_dg)

        sh = th * 5 + ch * 2 + ph * 3 + dh * 3
        sa = ta * 5 + ca * 2 + pa * 3 + da * 3
        return sh, sa, th, ta, ph, pa, dh, da
    
    def plot_params(self):
        self.plot_team_params(params = ["att","def_","dis_clean","dis_force","p_conv","int_try","int_pen","att_dg","int_dg","psi_logit_dg"])


class PoissonModel_latAtt(RugbyModel):
    def __init__(self, sd_scale = 1 ,scale_int = 0.5, **kwargs):
        super().__init__(**kwargs)    # run RugbyModel.__init__ first
        self.sd_scale = sd_scale
        self.scale_int = scale_int

    def _build(self, temporal_weights, data): # need a function to unpack data - can this be in rugyb model class?
        home_idx,away_idx,obs_tries_home,obs_tries_away,obs_pens_home,obs_pens_away,obs_conv_home,obs_conv_away,obs_points_home,obs_points_away ,obs_dgs_home,obs_dgs_away= super()._unpack_data(data)
        muTry = np.log(np.mean(np.concatenate([obs_tries_away, obs_tries_home])))
        muPen = np.log(np.mean(np.concatenate([obs_pens_away, obs_pens_home])))
        muConv = (obs_conv_home.sum() + obs_conv_away.sum()) / (obs_tries_home.sum() + obs_tries_away.sum())
        missing_home = data.loc[~data["home_team"].isin(self.team_to_idx), "home_team"].unique()
        missing_away = data.loc[~data["away_team"].isin(self.team_to_idx), "away_team"].unique()
        if len(missing_home) or len(missing_away):
            raise ValueError(f"Teams not in index — home: {missing_home}, away: {missing_away}")
        with pm.Model() as model:
            #home advantage - one for all teams
            home = pm.Normal("home",mu = 0, sigma = 0.3) 
            
            #Tries----

            sd_att=pm.HalfNormal("sd_att", sigma=self.sd_scale)#half normal
            sd_def=pm.HalfNormal("sd_def", sigma=self.sd_scale)#ha;f normal

            att = pm.ZeroSumNormal("att", sigma=sd_att, shape=self.n_teams)
            def_ = pm.ZeroSumNormal("def_", sigma=sd_def, shape=self.n_teams)

            int_try =  pm.Normal("int_try", mu=muTry, sigma=self.scale_int) #normal
            
            sigma_game = pm.HalfNormal("sigma_game", sigma=0.3)
            game_effect = pm.Normal(
                "game_effect",
                mu=0,
                sigma=sigma_game,
                shape=len(obs_tries_home)
            )

            lambda_home_tries = pm.math.exp(
                int_try
                + att[home_idx]
                - def_[away_idx]
                + home
                + game_effect
            )

            lambda_away_tries = pm.math.exp(
                int_try
                + att[away_idx]
                - def_[home_idx]
                + game_effect
            )

            ll_home_try = pm.logp(
                pm.Poisson.dist(mu=lambda_home_tries),
                obs_tries_home,
            )

            ll_away_try = pm.logp(
                pm.Poisson.dist(mu=lambda_away_tries),
                obs_tries_away,
            )
            pm.Potential("home_tries",   (temporal_weights * ll_home_try).sum())
            pm.Potential("away_tries", (temporal_weights * ll_away_try).sum())


            #pens----   

            sd_dis_clean = pm.HalfNormal("sd_dis_clean", sigma=self.sd_scale)#half normal
            sd_dis_force = pm.HalfNormal("sd_dis_force", sigma=self.sd_scale)#half normal


            dis_clean = pm.ZeroSumNormal("dis_clean",  sigma=sd_dis_clean, shape=self.n_teams)
            dis_force = pm.ZeroSumNormal("dis_force",  sigma=sd_dis_force, shape=self.n_teams)


            int_pen =  pm.Normal("int_pen", mu=muPen, sigma=self.scale_int) #normal

            lambda_home_pens =  pm.math.exp(int_pen+dis_force[home_idx]-dis_clean[away_idx]+home) 
            lambda_away_pens = pm.math.exp(int_pen+dis_force[away_idx]-dis_clean[home_idx]) 


            ll_home_pen= pm.logp(pm.Poisson.dist(mu = lambda_home_pens), obs_pens_home) 

            ll_away_pen = pm.logp(pm.Poisson.dist(mu = lambda_away_pens), obs_pens_away)

            pm.Potential("home_pens",   (temporal_weights * ll_home_pen).sum())
            pm.Potential("away_pens", (temporal_weights * ll_away_pen).sum())

            #conversions---

            beta_conv = pm.Beta("p_conv", mu=muConv, sigma=0.08,shape  = self.n_teams) # beta
            logp_conv_home = pm.logp(pm.Binomial.dist(n=obs_tries_home, p=beta_conv[home_idx]), obs_conv_home)
            logp_conv_away = pm.logp(pm.Binomial.dist(n=obs_tries_away, p=beta_conv[away_idx]), obs_conv_away)
            pm.Potential("home_conversions", (temporal_weights * logp_conv_home).sum()) # binomaial 
            pm.Potential("away_conversions", (temporal_weights * logp_conv_away).sum()) # binomaial
            self.model = model
            self.trace = pm.sample(self.nSamples//self.cores, tune=self.tune, cores=self.cores, target_accept = self.target_accept)
        self.unpack_posterior()
        
        
    def unpack_posterior(self):
        trace = self.trace
        
        n_samples = trace.posterior.dims["draw"] * trace.posterior.dims["chain"]

        def f(arr):
            return arr.values.reshape(n_samples, *arr.shape[2:])

        self.att       = f(trace.posterior["att"])
        self.def_      = f(trace.posterior["def_"])
        self.dis_clean = f(trace.posterior["dis_clean"])
        self.dis_force = f(trace.posterior["dis_force"])
        self.p_conv    = f(trace.posterior["p_conv"])
        self.int_try   = f(trace.posterior["int_try"])
        self.int_pen   = f(trace.posterior["int_pen"])
        self.home      = f(trace.posterior["home"])
        self.sigma_game = f(trace.posterior["sigma_game"])


    def predict_match(self, hi, ai,neutral = False):
        home = self.home if not neutral else 0
        gameEffect = np.random.normal(
            0,
            self.sigma_game.squeeze(),
            size=self.int_try.shape[0]
        )

        lam_h = np.exp(
            self.int_try
            + self.att[:, hi]
            - self.def_[:, ai]
            + self.home
            + gameEffect
        )

        lam_a = np.exp(
            self.int_try
            + self.att[:, ai]
            - self.def_[:, hi]
            + gameEffect
        )
        lam_h_pen = np.exp(self.int_pen + self.dis_force[:, hi] - self.dis_clean[:, ai] + home)
        lam_a_pen = np.exp(self.int_pen + self.dis_force[:, ai] - self.dis_clean[:, hi])
        th = np.random.poisson(lam_h);  ta = np.random.poisson(lam_a)
        ph = np.random.poisson(lam_h_pen); pa = np.random.poisson(lam_a_pen)
        ch = np.random.binomial(th, self.p_conv[:, hi])
        ca = np.random.binomial(ta, self.p_conv[:, ai])
        sh = th * 5 + ch * 2 + ph * 3
        sa = ta * 5 + ca * 2 + pa * 3
        return sh, sa, th, ta, ph, pa
    
    def plot_params(self):
        self.plot_team_params(params = ["att","def_","dis_clean","dis_force","p_conv","int_try","int_pen"])
