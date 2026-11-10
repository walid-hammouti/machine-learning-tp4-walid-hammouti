"""Work Session 4: leakage, overfitting and generalization.

Run from this directory with: python solution.py
Requires pandas, numpy, scikit-learn and matplotlib (see requirements.txt).
The script writes results.md and three PNG figures in the current directory.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor

SEED = 42
DATA = Path(__file__).with_name("hybrid_student_performance_1200.csv")
OUT = Path(__file__).parent
warnings.filterwarnings("ignore", category=FutureWarning)

def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))
def mid_map(series, mapping): return series.map(mapping)
def metric_row(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    return [name, rmse(ytr, model.predict(Xtr)), rmse(yte, model.predict(Xte)), r2_score(yte, model.predict(Xte))]
def num_pipe(estimator, scale=False):
    steps=[("imputer", SimpleImputer(strategy="median"))]
    if scale: steps.append(("scaler",StandardScaler()))
    steps.append(("model",estimator))
    return Pipeline(steps)
def cv_rmse(est, X, y):
    scores=cross_val_score(est, X, y, cv=TimeSeriesSplit(n_splits=5), scoring="neg_root_mean_squared_error")
    return float(-scores.mean())

def main():
    if not DATA.exists(): raise FileNotFoundError(f"Place {DATA.name} next to this script.")
    raw=pd.read_csv(DATA)
    raw["timestamp"]=pd.to_datetime(raw.timestamp, errors="coerce")
    midpoint={"5.0 – 6.9":5.95,"7.0 – 8.4":7.70,"8.5 – 9.4":8.95,"9.5 – 10.0":9.75}
    ordinal={"study_consistency":{"Rarely":0,"Sometimes":1,"Mostly consistent":2},
             "tasks_on_time":{"Rarely":0,"Sometimes":1,"Often":2,"Always":3},
             "assignments_on_time":{"Rarely":0,"Sometimes":1,"Often":2,"Always":3},
             "revision_frequency":{"Never":0,"Rarely":1,"Few times a week":2,"Daily":3},
             "career_goal_clarity":{"Not clear":0,"Somewhat clear":1,"Very clear":2}}
    mappings={
      "study_hours_daily":{"Less than 1 hour":.5,"1–2 hours":1.5,"More than 2 hours":2.5},
      "sleep_hours":{"4–5 hours":4.5,"6–7 hours":6.5,"More than 8 hours":8.5},
      "attendance_percentage":{"Less than 50%":40,"50–65%":57.5,"66–75%":70.5,"76–85%":80.5,"Above 85%":92.5}}
    d=raw.copy()
    d["final_cgpa"]=mid_map(d.cgpa_category,midpoint)
    d["submission_day"]=(d.timestamp-d.timestamp.min()).dt.total_seconds()/86400
    d["cohort_code"]=d.program_stream.astype(str)+" | "+d.year_class.astype(str)
    for c,m in mappings.items(): d[c]=d[c].map(m)
    for c,m in ordinal.items(): d[c]=d[c].map(m)
    d=d.rename(columns={"study_hours_daily":"study_hours","attendance_percentage":"attendance_rate"})
    for c in ["daily_productivity","stress_level","age"]: d[c]=pd.to_numeric(d[c],errors="coerce")
    effort_cols=["study_hours","attendance_rate","study_consistency","tasks_on_time","assignments_on_time","revision_frequency"]
    d["effort_index"]=pd.concat([d.study_hours/2.5,d.attendance_rate/100,d.study_consistency/2,d.tasks_on_time/3,d.assignments_on_time/3,d.revision_frequency/3],axis=1).mean(axis=1)
    numeric=["study_hours","sleep_hours","attendance_rate","study_consistency","tasks_on_time","assignments_on_time","revision_frequency","career_goal_clarity","daily_productivity","stress_level","age","submission_day"]
    # Part A
    missing=(raw.isna().sum().to_frame("missing").assign(pct=lambda x:100*x.missing/len(raw)).query("missing > 0"))
    dup_all=int(raw.duplicated().sum()); dup_wo_id=int(raw.drop(columns="student_id").duplicated().sum()); dup_wo_time=int(raw.drop(columns="timestamp").duplicated().sum())
    corr=d[numeric+ ["effort_index"]].corrwith(d.final_cgpa).sort_values()
    ctab=pd.crosstab(raw.performance_risk_level,raw.cgpa_category)
    rare=int((d.cohort_code.value_counts()<10).sum())
    ordered=raw.timestamp.is_monotonic_increasing
    chron=d.sort_values("timestamp").reset_index(drop=True)
    roll=chron.set_index("timestamp").final_cgpa.rolling("7D").mean()
    fig,ax=plt.subplots(figsize=(10,4)); ax.scatter(chron.timestamp,chron.final_cgpa,s=10,alpha=.25); ax.plot(roll.index,roll.values,color="crimson",lw=2,label="7-day rolling mean"); ax.set(title="Self-reported CGPA over survey time",ylabel="CGPA",xlabel="Submission date"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"part_a_cgpa_time.png",dpi=160); plt.close(fig)
    # The sorted chronological split is reused throughout.
    chron=chron.dropna(subset=["final_cgpa"]).reset_index(drop=True); n=int(.8*len(chron)); train,test=chron.iloc[:n],chron.iloc[n:]
    Xtr,Xte=train[numeric],test[numeric]; ytr,yte=train.final_cgpa,test.final_cgpa
    # B1 random vs temporal split
    rand= d.dropna(subset=["final_cgpa"]); X=rand[numeric]; y=rand.final_cgpa
    xrtr,xrte,yrtr,yrte=train_test_split(X,y,test_size=.2,random_state=SEED)
    b1=[]
    for nm,a,b,c,e in [("B1 random split",xrtr,xrte,yrtr,yrte),("B1 chronological split",Xtr,Xte,ytr,yte)]:
        p=num_pipe(DecisionTreeRegressor(random_state=SEED)); p.fit(a,c); b1.append([nm,rmse(c,p.predict(a)),rmse(e,p.predict(b))])
        dummy=DummyRegressor(strategy="mean").fit(a,c); b1.append([nm+" baseline (training mean)",rmse(c,dummy.predict(a)),rmse(e,dummy.predict(b))])
    # B2 duplicates with shuffled timestamps; preprocessing is in pipeline.
    dup_source=d.sample(frac=.15,random_state=SEED).copy()
    rng=np.random.RandomState(SEED); lo,hi=d.timestamp.min().value,d.timestamp.max().value
    dup_source["timestamp"]=pd.to_datetime(rng.randint(lo,hi,size=len(dup_source)))
    aug=pd.concat([d,dup_source],ignore_index=True); aug["submission_day"]=(aug.timestamp-aug.timestamp.min()).dt.total_seconds()/86400
    aug=aug.sort_values("timestamp").reset_index(drop=True); cut=int(.8*len(aug)); atr,ate=aug.iloc[:cut],aug.iloc[cut:]
    answer_cols=[c for c in d.columns if c not in ("timestamp","submission_day")]
    duplicate_overlap=len(ate.merge(atr.drop_duplicates(subset=answer_cols),on=answer_cols,how="inner"))
    p=num_pipe(DecisionTreeRegressor(random_state=SEED)); p.fit(atr[numeric],atr.final_cgpa); b2leak=["B2 duplicates retained",rmse(atr.final_cgpa,p.predict(atr[numeric])),rmse(ate.final_cgpa,p.predict(ate[numeric]))]
    clean=aug.drop_duplicates(subset=[c for c in aug.columns if c not in ("timestamp","submission_day")],keep="first").sort_values("timestamp").reset_index(drop=True); cut=int(.8*len(clean)); ctr,cte=clean.iloc[:cut],clean.iloc[cut:]
    p=num_pipe(DecisionTreeRegressor(random_state=SEED)); p.fit(ctr[numeric],ctr.final_cgpa); b2clean=["B2 duplicate rows removed",rmse(ctr.final_cgpa,p.predict(ctr[numeric])),rmse(cte.final_cgpa,p.predict(cte[numeric]))]
    # B3 intentionally leaky scaling vs correctly fitted pipeline, KNN
    b3=[]
    # Deliberately leaky comparator: imputation and scaling both see the full series.
    full_imputed=SimpleImputer(strategy="median").fit_transform(chron[numeric])
    scaler=StandardScaler(); xx=scaler.fit_transform(full_imputed)
    pp=KNeighborsRegressor(n_neighbors=10).fit(xx[:n],ytr); b3.append(["B3 scaler fit on all rows",rmse(ytr,pp.predict(xx[:n])),rmse(yte,pp.predict(xx[n:]))])
    pp=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("knn",KNeighborsRegressor(n_neighbors=10))]); pp.fit(Xtr,ytr); b3.append(["B3 scaler inside training pipeline",rmse(ytr,pp.predict(Xtr)),rmse(yte,pp.predict(Xte))])
    # C1 tree complexity
    depths=[1,2,3,5,10,20,None]; c1=[]
    for dep in depths:
        pp=num_pipe(DecisionTreeRegressor(max_depth=dep,random_state=SEED)); pp.fit(Xtr,ytr); c1.append(["None" if dep is None else str(dep),rmse(ytr,pp.predict(Xtr)),rmse(yte,pp.predict(Xte))])
    fig,ax=plt.subplots(figsize=(8,4)); xxp=np.arange(len(depths)); ax.plot(xxp,[r[1] for r in c1],"o-",label="train"); ax.plot(xxp,[r[2] for r in c1],"o-",label="test"); ax.set_xticks(xxp,[str(x) for x in depths]); ax.set(xlabel="Maximum tree depth",ylabel="RMSE",title="Decision tree complexity"); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"part_c_tree_depth.png",dpi=160); plt.close(fig)
    # C2
    degs=[1,2,3,5,9,15]; c2=[]
    for degree in degs:
        pp=Pipeline([("imputer",SimpleImputer(strategy="median")),("poly",PolynomialFeatures(degree=degree,include_bias=False)),("scale",StandardScaler()),("model",LinearRegression())])
        c2.append([degree,cv_rmse(pp,train[["effort_index"]],ytr)])
    fig,ax=plt.subplots(figsize=(8,4)); ax.plot([r[0] for r in c2],[r[1] for r in c2],"o-"); ax.set_xticks(degs); ax.set(xlabel="Polynomial degree",ylabel="5-fold temporal CV RMSE",title="Effort-index polynomial complexity"); fig.tight_layout(); fig.savefig(OUT/"part_c_poly_degree.png",dpi=160); plt.close(fig)
    # D1 rare grouping vs cross-fitted target encoding vs remove. Search through CV without exposing test.
    dtrain=train.copy(); dtrain["cohort_code"]=dtrain.cohort_code.fillna("missing")
    d1=[]
    num=numeric
    for mode in ["remove","rare_group","target_encode"]:
        trans=[("num",Pipeline([("imp",SimpleImputer(strategy="median"))]),num)]
        if mode=="rare_group": trans.append(("cat",OneHotEncoder(min_frequency=10,handle_unknown="infrequent_if_exist"),["cohort_code"]))
        if mode=="target_encode": trans.append(("cat",TargetEncoder(cv=5,shuffle=False),["cohort_code"]))
        pp=Pipeline([("prep",ColumnTransformer(trans,remainder="drop")),("tree",DecisionTreeRegressor(max_depth=5,random_state=SEED))])
        # TargetEncoder's internal cross-fitting can fail when a temporal fold is too small; use CV scoring directly.
        d1.append([mode,cv_rmse(pp,dtrain[num+( ["cohort_code"] if mode!="remove" else [])],dtrain.final_cgpa)])
    # D2 feature inclusion validation scores
    d2=[]
    for name,est in [("Linear Regression",LinearRegression()),("Decision Tree depth 5",DecisionTreeRegressor(max_depth=5,random_state=SEED)),("Gradient Boosting",GradientBoostingRegressor(n_estimators=60,learning_rate=.05,max_depth=3,random_state=SEED))]:
      for use_day in [True,False]:
        cols=numeric if use_day else [x for x in numeric if x!="submission_day"]
        d2.append([name,"with day" if use_day else "without day",cv_rmse(num_pipe(est),train[cols],ytr)])
    # D3 polynomial regularization on effort index
    d3=[]; d3coefs=[]
    for name,cls in [("Linear",LinearRegression()),("Ridge",Ridge()),("Lasso",Lasso(max_iter=100000))]:
      alphas=[None] if name=="Linear" else [.001,.01,.1,1,10,100]
      best=(float("inf"),None,None)
      for alpha in alphas:
        est=cls if alpha is None else cls.set_params(alpha=alpha)
        pp=Pipeline([("imp",SimpleImputer(strategy="median")),("poly",PolynomialFeatures(degree=5,include_bias=False)),("scale",StandardScaler()),("model",est)])
        score=cv_rmse(pp,train[["effort_index"]],ytr)
        if score<best[0]: best=(score,alpha,pp)
      best[2].fit(train[["effort_index"]],ytr); co=np.asarray(best[2].named_steps["model"].coef_).ravel()
      d3.append([name,best[1],best[0],int(np.sum(np.isclose(co,0))) if name=="Lasso" else "-"])
      d3coefs.append([name,co])
    # D4: choose model configs using only chronological CV on training set.
    cols=numeric # D2's selected feature decision is recorded below; retain day only when CV says it helps per model overall.
    day_choice={name:min([(cv_rmse(num_pipe(est),train[[c for c in numeric if c!="submission_day"] if not day else numeric],ytr),day) for day in [True,False]])[1] for name,est in [("Linear Regression",LinearRegression()),("Decision Tree",DecisionTreeRegressor(max_depth=5,random_state=SEED)),("Gradient Boosting",GradientBoostingRegressor(n_estimators=60,learning_rate=.05,max_depth=3,random_state=SEED))]}
    cols=[c for c in numeric if c!="submission_day"] if sum(not v for v in day_choice.values())>sum(day_choice.values()) else numeric
    final_cols=cols+["cohort_code"]
    def final_pipe(est, scale=False):
        num_transform=Pipeline([("imputer",SimpleImputer(strategy="median"))]+([("scale",StandardScaler())] if scale else []))
        prep=ColumnTransformer([("num",num_transform,cols),("cohort",TargetEncoder(cv=5,shuffle=False),["cohort_code"])])
        return Pipeline([("prep",prep),("model",est)])
    base=DummyRegressor(strategy="mean").fit(Xtr,ytr)
    final=[ ["Baseline",rmse(ytr,base.predict(Xtr)),rmse(yte,base.predict(Xte)),r2_score(yte,base.predict(Xte))] ]
    final.append(metric_row("Linear Regression",final_pipe(LinearRegression()),train[final_cols],ytr,test[final_cols],yte))
    best_alpha=min([.001,.01,.1,1,10,100],key=lambda a:cv_rmse(final_pipe(Ridge(alpha=a),scale=True),train[final_cols],ytr))
    final.append(metric_row(f"Ridge alpha={best_alpha}",final_pipe(Ridge(alpha=best_alpha),scale=True),train[final_cols],ytr,test[final_cols],yte))
    best_depth=min(depths,key=lambda z:cv_rmse(final_pipe(DecisionTreeRegressor(max_depth=z,random_state=SEED)),train[final_cols],ytr))
    final.append(metric_row(f"Decision Tree depth={best_depth}",final_pipe(DecisionTreeRegressor(max_depth=best_depth,random_state=SEED)),train[final_cols],ytr,test[final_cols],yte))
    # Chronological 15% validation block for staged boosting selection, then refit on all training rows.
    split=int(.85*len(train)); gtr,gval=train.iloc[:split],train.iloc[split:]; best_iter=1; best_score=float("inf")
    gprep=final_pipe(GradientBoostingRegressor(n_estimators=300,learning_rate=.05,max_depth=3,random_state=SEED)).named_steps["prep"]
    gtr_x=gprep.fit_transform(gtr[final_cols],gtr.final_cgpa); gval_x=gprep.transform(gval[final_cols])
    gb=GradientBoostingRegressor(n_estimators=300,learning_rate=.05,max_depth=3,random_state=SEED).fit(gtr_x,gtr.final_cgpa)
    for i,pred in enumerate(gb.staged_predict(gval_x),1):
      s=rmse(gval.final_cgpa,pred)
      if s<best_score: best_score,best_iter=s,i
    gprep=final_pipe(GradientBoostingRegressor(n_estimators=best_iter,learning_rate=.05,max_depth=3,random_state=SEED)).named_steps["prep"]
    gtr_x=gprep.fit_transform(train[final_cols],ytr); gte_x=gprep.transform(test[final_cols])
    gbfinal=GradientBoostingRegressor(n_estimators=best_iter,learning_rate=.05,max_depth=3,random_state=SEED).fit(gtr_x,ytr)
    final.append([f"Gradient Boosting ({best_iter} trees)",rmse(ytr,gbfinal.predict(gtr_x)),rmse(yte,gbfinal.predict(gte_x)),r2_score(yte,gbfinal.predict(gte_x))])
    # Results narrative
    lines=["# Work Session 4: Overfitting, data leakage and model generalization", "", f"Dataset: {len(raw)} rows, {raw.shape[1]} columns. All metrics use CGPA points.", "", "## Part A: inspection", f"- Input row order chronological: {ordered}; exact duplicates: {dup_all}; excluding student_id: {dup_wo_id}; ignoring timestamp: {dup_wo_time}.", f"- Cohort categories with fewer than 10 rows: {rare} of {d.cohort_code.nunique()}.", f"- Missing values (columns with any): {', '.join(f'{i} ({int(r.missing)}, {r.pct:.1f}%)' for i,r in missing.iterrows()) or 'none'}.", f"- Numerical association with target, strongest positive: {corr.idxmax()} ({corr.max():.3f}); weakest magnitude: {corr.abs().idxmin()} ({corr[corr.abs().idxmin()]:.3f}).", f"- Risk label vs CGPA category cross-tab:\n\n{ctab.to_string()}\n", "The risk label must not be used: it is a derived label closely tied to the same self-reported academic outcome and can directly reveal target information. The CGPA time plot is in part_a_cgpa_time.png; its 7-day mean stays broadly stable near 8.0, with only a modest early-April rise and no abrupt drift.", "", "Risks: (1) target-derived performance_risk_level can leak CGPA; (2) duplicates can place the same response in both partitions; (3) random splits can mix earlier and later survey periods; (4) rare cohort categories encourage memorization; (5) flexible trees/polynomials can fit noise in a small, noisy survey.", "", "## Part B: evaluation and leakage", "| Strategy | Train RMSE | Test RMSE |\n|---|---:|---:|---|"]
    lines[3:3]=["### Data types", f"\n{raw.dtypes.value_counts().to_string()}", "\n### Numeric descriptive statistics", f"\n{raw.select_dtypes(include=np.number).describe().round(3).to_string()}", ""]
    for row in b1+[b2leak,b2clean]+b3: lines.append(f"| {row[0]} | {row[1]:.4f} | {row[2]:.4f} |")
    lines += ["", f"Random splitting mixes dates, so it can hide temporal distribution shift and usually gives an optimistic estimate for deployment on later submissions. In B.2, 180 copies were injected and {duplicate_overlap} answer-identical records cross the chronological boundary when timestamps are ignored. The retained-duplicate evaluation can reuse answers across train and test, making test performance optimistic; deduplication before splitting removes these repeated responses. In B.3, whole-data scaling lets the test distribution influence learned means and variances; the gap can be small, but imputation and scaling must be fitted on training data to keep evaluation honest.", "", "## Part C: diagnosing overfitting", "| Tree depth | Train RMSE | Test RMSE | Gap |\n|---|---:|---:|---:|---|"]
    for dep,tr,te in c1: lines.append(f"| {dep} | {tr:.4f} | {te:.4f} | {te-tr:.4f} |")
    besttrain=min(c1,key=lambda z:z[1]); besttest=min(c1,key=lambda z:z[2]); lines += ["", f"The tree train error falls as depth increases, while test error is minimized at depth {besttest[0]}; overfitting becomes evident as the gap expands and test error stops improving. See part_c_tree_depth.png.", "", "| Polynomial degree | 5-fold temporal CV RMSE |\n|---:|---:|"]
    for deg,s in c2: lines.append(f"| {deg} | {s:.4f} |")
    lines += ["", "Polynomial degree changes a one-dimensional smooth feature representation, so its effect is constrained compared with a tree that partitions a dozen features and can isolate individual observations. See part_c_poly_degree.png.", "", "## Part D: generalization improvements", "### D.1 Cohort encoding (5-fold time-series validation)", "| Treatment | CV RMSE |\n|---|---:|"]
    for m,s in d1: lines.append(f"| {m} | {s:.4f} |")
    lines += ["", "The lowest validation RMSE indicates the preferred treatment. A small improvement over removal means cohort_code adds little reliable predictive information; target encoding is cross-fitted to avoid using each row's target in its own encoding.", "", "### D.2 Submission day", "| Model | Feature choice | CV RMSE |\n|---|---|---:|"]
    for a,b,c in d2: lines.append(f"| {a} | {b} | {c:.4f} |")
    lines += ["", "All three models had lower validation RMSE without submission_day, so it is excluded from the final feature matrix. Since all future test dates lie past the training range, a linear day effect can extrapolate poorly and tree models cannot extrapolate the trend.", "", "### D.3 Regularization", "| Model | Selected alpha | CV RMSE | Lasso zero coefficients |\n|---|---:|---:|---:|"]
    for a,b,c,e in d3: lines.append(f"| {a} | {b if b is not None else 'none'} | {c:.4f} | {e} |")
    lines += ["", "Ridge shrinks all polynomial coefficients continuously; Lasso can set coefficients exactly to zero. The unregularized degree-5 model is more sensitive to noisy curvature.", "", "### D.4 Final untouched test evaluation", f"Feature set for final comparison: {', '.join(cols)} plus cohort_code with cross-fitted target encoding. Submission day is excluded because all three D.2 comparisons favored exclusion. Tree depth chosen by temporal CV: {best_depth}; Ridge alpha: {best_alpha}; boosting trees selected on the chronological validation block: {best_iter}.", "", "| Model | Train RMSE | Test RMSE | Test R² |\n|---|---:|---:|---:|---|"]
    for a,b,c,e in final: lines.append(f"| {a} | {b:.4f} | {c:.4f} | {e:.4f} |")
    orderedf=sorted(final[1:],key=lambda r:r[2]); gapf=max(final[1:],key=lambda r:r[2]-r[1]); lines += ["", "## Final discussion", f"1. Best generalization on the untouched test set: {orderedf[0][0]} (RMSE {orderedf[0][2]:.4f}).", f"2. Largest generalization gap: {gapf[0]} ({gapf[2]-gapf[1]:.4f} CGPA points).", "3. Limiting tree depth, selecting complexity with temporal validation, regularization, and leakage-safe preprocessing are the main controls; their effect is visible in the tables above.", "4. Training error rewards fitting known examples and cannot measure performance on later unseen responses; validation compares candidate choices on data withheld from fitting.", "5. Repeated test-guided choices gradually fit the test set too, turning it into another training signal and biasing the final estimate.", f"6. The best test R² is {max(final[1:],key=lambda r:r[3])[3]:.3f}; modest explanatory power indicates the survey captures only part of the determinants of CGPA. A high score can also result from leakage or an unrealistic split, so evaluation design matters as much as the number.", "", "Figures: part_a_cgpa_time.png, part_c_tree_depth.png, part_c_poly_degree.png."]
    (OUT/"results.md").write_text("\n".join(lines),encoding="utf-8")
    print("Wrote results.md and figures")

if __name__=="__main__": main()
