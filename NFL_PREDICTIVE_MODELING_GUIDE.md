# A Technical Treatise on NFL Predictive Modeling

**Research by Google Gemini**

## Part I: The Architecture of the NFL Betting Market

A predictive model, regardless of its statistical sophistication, does not operate in a vacuum. It competes within a dynamic, information-driven financial market. To build a successful NFL betting model, one must first deconstruct the architecture of this market, understand the incentives of its primary actors (the bookmakers), and be able to decode the information it disseminates. This section provides a technical analysis of the market structure, price discovery mechanisms, and mathematical principles that govern NFL betting.

### Section 1.1: Deconstructing the Opening Line

The process of "setting a line" is widely misunderstood. The "opening line" is not a static, universally agreed-upon prediction of a game's outcome.¹ Rather, it is the initial hypothesis in a week-long price discovery process, originated by a small subset of "market-making" sportsbooks.²

These market-making oddsmakers, or "sports traders"³, employ sophisticated computer programs and proprietary power ratings, augmented by situational adjustments for injuries, weather, and home-field advantage, to generate this initial number.⁴ A critical distinction exists between these market-makers (e.g., Circa, Pinnacle, Bookmaker)⁶ and the majority of "line-following" or "soft" books, which simply copy the market-makers' lines.

The opening line is released in a deliberate, four-step process.² It is first posted with very low betting limits. This is not an act of caution; it is an invitation. This low-limit "feeler" line is an invitation to the world's most "respected professional bettors" (i.e., "sharps") to audit the number.¹ The market-maker wants these professional syndicates to "hammer" a line they deem inaccurate.¹

This initial sharp action provides the bookmaker with invaluable, free information. The bookmaker is not, therefore, "setting the line"; they are discovering the line.¹ The sharp bettors, by correcting the opener, dictate what the true, efficient line should be.¹ Once this line "settles" after absorbing this initial wave of professional money, the market-maker raises the limits, and other sportsbooks across the globe copy the now-efficient number.² This initial professional action is a "strong indication that number was off".⁴

For a quantitative modeler, this process has profound implications. The goal is not simply to "beat the opener," which is often intentionally soft. The goal is to develop a model that can identify value before the market-making sharps do, and ultimately, to possess a predictive accuracy that can find value against the "closing line"—the market's most efficient, fully-informed price.

### Section 1.2: The "Balancing the Book" Fallacy

A foundational misconception in sports betting is that a bookmaker's objective is to "balance the book"—that is, to attract an equal amount of money (handle) on both sides of a wager.⁹ In this mythical scenario, the bookmaker would guarantee a risk-free profit by paying the winners with the losers' money and keeping the commission, or "vigorish".¹²

This concept is, by the admission of numerous longtime Las Vegas bookmakers, "essentially fiction".¹³ It has been described as an "urban legend" and a "fallacy".¹³ The primary reasons for this are twofold:

1. **It is operationally impossible**: The complex interplay of parlays, teasers, and moneyline bets makes true 50/50 balancing of liability on every single game an operational impossibility.¹³

2. **It is not the true objective**: The bookmaker's actual goal is not risk elimination but profit maximization.¹⁴

A modern sportsbook operates as a risk manager with highly sophisticated in-house models.¹⁴ They "take a position" on a game based on what their own quantitative analysis indicates the line should be.¹³ If the book's models and the action from respected sharp bettors align on one side, the book is more than happy to accept lopsided "public" money on the other.¹² They are not "balancing" this public action; they are taking a position against it, confident that they are in an "advantageous position".¹⁶

This reframes the entire task of predictive modeling. The sports betting ecosystem is not a simple casino game where the house collects a fee. It is a predatory financial market where the bookmaker acts as a sophisticated, well-capitalized counterparty.¹² The bookmaker is actively betting against the uninformed public. A model, therefore, must be robust enough to identify lines where the bookmaker's position (informed by their own models and sharp action) is incorrect.

### Section 1.3: Vigorish, Implied Probability, and "True Odds"

To identify an incorrect line, a model must first be able to translate the market's price (the odds) into its "true" or "no-vig" probability.

#### The Vigorish (Vig):

The vigorish, or "juice," is the bookmaker's commission for facilitating a bet.¹⁷ In standard NFL spreads and totals, both sides are offered at -110 odds. This requires a bettor to risk $110 to win $100.¹⁸ If two bettors place opposing $110 wagers ($220 total handle), the bookmaker pays the winner $210 (their $100 profit + $110 stake) and keeps the remaining $10 from the loser's stake.¹⁷ In this -110/-110 scenario, the vig is 4.54% ($10 / $220).¹⁷

#### Implied Probability (IP):

Implied probability is the probability of an outcome as implied by the odds, which includes the vigorish. This calculation is non-negotiable for any betting model.

**Formula for Negative (Favorite) Odds:**

$$IP = \frac{|Odds|}{(|Odds| + 100)}$$

Example (-110 odds): $IP = \frac{110}{(110 + 100)} = \frac{110}{210} = 52.38\%$

**Formula for Positive (Underdog) Odds:**

$$IP = \frac{100}{(Odds + 100)}$$

Example (+120 odds): $IP = \frac{100}{(120 + 100)} = \frac{100}{220} = 45.45\%$

#### The "Overround" and Deriving "True" Odds:

Because the vig is included, the implied probabilities of all outcomes in a market will sum to more than 100%. This surplus is the "overround".²² For a -110/-110 market, the overround is $52.38\% + 52.38\% = 104.76\%$.²³ The 4.76% surplus is the bookmaker's total margin.

To evaluate a model, the vig must be removed to find the "fair" or "true" probability of each outcome.

**Formula for "True" (No-Vig) Probability:**

$$True \, Probability = \frac{Implied \, Probability \, of \, Outcome}{Total \, Implied \, Probability \, of \, All \, Outcomes}$$

Example (-110/-110 market):

$True \, P(Team \, A) = \frac{52.38\%}{104.76\%} = 50.0\%$

This confirms the market's "true" price on a -110 line is 50/50.²³

This conversion is the "Rosetta Stone" for a betting model. The model's code will not output "-110"; it will output a true probability (e.g., "Team A has a 55% chance to cover"). The formulas above are required to convert the market's price (-110) into a comparable true probability (50.0%). The model's core function is this comparison: $55\% > 50.0\%$. This difference identifies a positive expected value (+EV) bet.

The 52.38% (often rounded to 52.4%) figure is the actual win rate a bettor must achieve at -110 odds just to break even against the vig.²⁴

#### Table 1: Implied Probability & "No-Vig" Conversion Table

| Bet Type | American Odds | Implied Probability (IP) | No-Vig "True" Probability (Break-Even %) |
|----------|---------------|--------------------------|------------------------------------------|
| Standard Spread/Total | -110 | 52.38% | 50.0% (Break-even: 52.4%) |
| Standard Spread/Total | -110 | 52.38% | 50.0% |
| **Sum of Market** | | **104.76%** | **100.0%** |
| Favorite | -150 | 60.00% | 52.9% |
| Underdog | +130 | 43.48% | 47.1% |
| **Sum of Market** | | **103.48%** | **100.0%** |
| Favorite | -200 | 66.67% | 54.5% |
| Underdog | +175 | 36.36% | 45.5% |
| **Sum of Market** | | **103.03%** | **100.0%** |

### Section 1.4: Decoding Market-Driven Information: Line Movement

The betting line is not static; it is a living "price" that moves in response to new information. This information comes from two primary sources: "public money" and "sharp money".²⁸ Public money represents the (often biased) wagers of the general public, who tend to favor favorites and home teams.¹² "Sharp money" represents the high-volume, respected wagers from professional syndicates.¹

The consensus among market-makers and analysts is that sharp money moves lines.¹ A flood of public money may cause a book to adjust a line to manage liability (e.g., moving a favorite from -7 to -7.5 to entice underdog bets), but this is a risk management move.²⁰ Conversely, a single respected sharp bet can cause an immediate, defensive line move as the book respects their opinion and adjusts their "position".⁴

A common proxy for tracking sharp action is to compare the percentage of bets (tickets) to the percentage of money (handle).²⁸ A "low bets, higher dollars" split, such as a team receiving 64% of bets but 90% of the money, indicates that while the public is slightly on that side, the large, professional wagers are also on that side.³² A more dramatic split, such as 25% of bets accounting for 70% of the money, is a classic signal that the public is on one side while the sharps are heavily on the other.³³

#### Reverse Line Movement (RLM):

This is the market's most powerful information signal.³³ RLM occurs when the majority of public bets are on one side, but the betting line moves in the opposite direction.³¹

**Example**: 75% of public bets are on the New England Patriots -3. The line, however, moves down to Patriots -2.5.³³

This phenomenon is a direct refutation of the "balancing the book" myth. If a bookmaker were balancing, they would move the line to -3.5 to attract bets on the underdog. By moving to -2.5, they are decreasing the price on the publicly-backed Patriots and worsening the number for the underdog. This is a defensive reaction. It signals that respected sharp money has bet the underdog at +3, and the bookmaker is "fleeing" to a new number (e.g., +2.5) to avoid taking more sharp action.³⁵

For a quantitative modeler, this is a critical, real-time data point. A model's code should incorporate a "Market Monitor" module. If the model's prediction identifies "Patriots -3" as a +EV bet, but the Market Monitor detects RLM against the Patriots, this is a strong external signal that the market's sharpest participants disagree with the model. This signal can be used to flag the bet as high-risk or to trigger an automatic "no-bet" decision.

## Part II: Data Sourcing and Feature Engineering (The Model's "Brain")

A predictive model is fundamentally a system for processing information. The quality and predictive power of the model are, therefore, entirely dependent on the quality of its input data and the sophistication of its "feature engineering"—the process of transforming raw data into predictive variables.³⁹

### Section 2.1: Acquiring the Raw Material: A Guide to Data Sources

A robust model requires three categories of data: play-by-play (PBP) statistics, advanced team metrics, and historical betting odds.

#### 1. Play-by-Play (PBP) Data:

This is the gold standard for modern sports analytics.

**Source**: **nflfastR**.⁴² This R package (with data easily accessible in Python as well) is the undisputed open-source standard. It provides clean, regularly updated play-by-play data for every NFL game, dating back to 1999.⁴⁴

**Content**: nflfastR is not merely a data scraper; it is a complete analytics package. The PBP data comes pre-loaded with the most critical advanced metrics, including Expected Points Added (EPA), Win Probability (WP), and Completion Probability Over Expected (CPOE).⁴³ This makes it an all-in-one data source for modeling on-field action.

#### Table 2: Key nflfastR Data Fields for a Betting Model

| Field Name | Description | Model Application |
|------------|-------------|-------------------|
| `game_id` | Unique game identifier. | Primary key for joining all data. |
| `posteam` | Team in possession (offense). | Grouping by offensive team. |
| `defteam` | Team on defense. | Grouping by defensive team. |
| `down` | The current down (1-4). | Critical input for Expected Points model. |
| `ydstogo` | Yards needed for a first down. | Critical input for Expected Points model. |
| `yardline_100` | Numeric distance from opponent's endzone. | Critical input for Expected Points model. |
| `game_seconds_remaining` | Total seconds left in the game. | Critical input for Win Probability model. |
| `score_differential` | posteam score minus defteam score. | Critical input for Win Probability model. |
| `epa` | Expected Points Added on the play. | The single most important predictive variable. |
| `cpoe` | Completion Probability Over Expected. | Measures quarterback accuracy vs. expectation. |
| `pass` | Binary (1/0) if a pass play. | Filtering for pass vs. rush efficiency. |
| `rush` | Binary (1/0) if a rush play. | Filtering for pass vs. rush efficiency. |
| `spread_line` | The game's closing point spread. | The target variable for backtesting ATS. |
| `total_line` | The game's closing over/under line. | The target variable for backtesting totals. |

This database schema is the blueprint for the model. The model's code will be built to load this data (e.g., `nflfastR::load_pbp(seasons = 2018:2023)`)⁴⁶ and then aggregate it (e.g., `group_by(posteam)` and `summarise(avg_epa = mean(epa, na.rm = TRUE))`).⁴³

#### 2. Advanced Team Metrics:

**Source (DVOA)**: Defense-adjusted Value Over Average (DVOA) is an advanced metric created by Football Outsiders⁴⁸, which is now hosted by FTN.⁴⁹ This metric is highly predictive but, unlike EPA, historical DVOA data is often proprietary and may require a paid subscription.⁵²

#### 3. Historical Betting Odds:

**Source (Free)**: This is the most difficult data to acquire reliably for free. nflfastR includes closing lines, which is sufficient for backtesting.⁵⁴ Other sources for historical data include RotoWire²⁴ and Kaggle datasets.⁵⁶

**Source (Paid)**: A professional-grade, real-time model requires a paid odds API. Services like SportsDataIO⁵⁷, OpticOdds⁵⁸, OddsMatrix⁵⁹, OddsJam⁶⁰, and The Odds API⁶¹ provide live and historical odds feeds from hundreds of sportsbooks.

A highly effective "version 1.0" model can be built entirely using the free nflfastR dataset. Its inclusion of `epa` (for prediction) and `spread_line` (for backtesting) makes it a self-contained ecosystem.

### Section 2.2: Feature Engineering I: Quantifying Contextual Variables

Feature engineering¹⁷¹ is what separates a generic model from a sharp one. It involves encoding domain-specific knowledge (e.g., the effect of weather) into a numerical format the model can understand.

#### 1. Home-Field Advantage (HFA):

**The Myth**: HFA is a static 3-point advantage.⁶²

**The Reality**: HFA is a dynamic and decaying variable. While historically valued at 2.5-3 points⁶⁴, its value has steadily trended downward. In the modern, post-COVID era, the true league-wide average is closer to 1.5 points.⁶² Some analyses place it between 2.2 and 2.6 points⁶⁴, but the 3-point standard is obsolete.

**Implementation**: A model must not use a static 3-point HFA. A simple model should use a dynamic, recent average (e.g., 1.5 points). A more advanced model would create team-specific HFAs, recognizing that factors like altitude (Denver) or crowd noise (Seattle) may create unique advantages.⁶⁶

#### 2. Quantifying Weather:

This is a critical, and often mis-quantified, variable.

**The Key Factor**: Wind speed is the single most impactful weather element, far more so than rain or cold.⁶⁷

**Wind**: The effect is non-linear.

- **10-15 mph**: "Noticeable decline" in passing.⁶⁹
- **15-20 mph**: "Significant drop" in passing production and efficiency.⁶⁹
- **20+ mph**: This is the critical threshold. It causes a "severe decline" in passing, efficiency, and scoring, and a "massive discrepancy" in field goal success. At this speed, FG conversion rates drop ~6% below what would be expected, even on shorter-than-average attempts.⁶⁹

**Precipitation**:

- **Rain**: Can lower a quarterback's completion percentage by ~12%⁷⁰ and leads to an average of 2-6 fewer total points.⁷⁰
- **Heavy Snow**: Has a "significant impact," decreasing passing production by ~25%⁶⁹ and reducing total scoring by as much as 10 points.⁷⁰

**Implementation**: The model's code must ingest game-day weather forecasts²⁴ and apply a numerical adjustment. The following table provides a heuristic framework for creating this `weather_adjustment` feature.

#### Table 3: NFL Weather Impact Heuristics (for Model Adjustment)

| Condition | Wind Speed | Precipitation | Expected Impact on Game | Model Adjustment (Total) |
|-----------|------------|---------------|-------------------------|--------------------------|
| Clear | 0-10 mph | None | None. Baseline conditions. | 0.0 points |
| Windy | 15-20 mph | None | Noticeable drop in passing & kicking. | -1.5 points |
| Very Windy | 20+ mph | None | "Severe decline" in passing. "Massive discrepancy" in FG success. | -4.0 points |
| Rain | < 15 mph | Moderate | "Lower throw accuracy." Run game favored. | -2.5 points |
| Wind + Rain | > 15 mph | Moderate | "Complicated and messy." Run-first scripts. | -5.5 points |
| Heavy Snow | Any | Heavy | "Significant impact." -25% passing. | -7.0 to -10.0 points |

This table allows a modeler to quantify a subjective variable. By scraping weather data, a `weather_adjustment` feature can be created, allowing the model to make situational, contextual predictions that simple team-vs-team models cannot.⁷³

#### 3. Quantifying Injuries:

This is the most difficult feature to engineer, as all injuries are not weighted equally.⁷⁵

**Positional Value**: The model must understand positional value.

- **Quarterback (QB)**: This is the most valuable position by an order of magnitude. The absence of a star QB (e.g., Josh Allen) can shift a line by 6 or more points.⁷⁶ The drop-off from a starter to a backup is the single largest adjustment an oddsmaker makes.⁷⁵
- **Other Key Positions**: Injuries to the offensive line (O-Line), particularly Left Tackle, are the next most critical, as they impact both passing and rushing efficiency.⁶² Elite pass rushers and "shutdown" cornerbacks also carry significant, quantifiable value.⁷⁶

**Implementation**: A simple model can create a "QB adjustment" feature based on the starter's status (e.g., using `qbelo1_pre`, a QB-adjusted Elo rating).⁷⁹ A more advanced model⁸⁰ would create a "player rating" system (e.g., using a Wins Above Replacement, or WAR, metric) and assign a "points to the spread" value for every starter. The sum of injured players' values becomes the `injury_adjustment` feature.

### Section 2.3: Feature Engineering II: Advanced Analytical Metrics

The core of a modern predictive model is the rejection of traditional box score statistics (like "total yards per game" or "time of possession") in favor of efficiency metrics that measure play-by-play value.⁸¹

#### 1. Expected Points Added (EPA):

**Definition**: EPA is the fundamental metric of play-by-play value. It measures the change in Expected Points (EP) from the start of a play to the end of that play.⁸³ Expected Points is the number of points a team is expected to score (on average) from a specific down, distance, and field position.

**Example**: A team starts a drive 1st-and-10 from their own 25-yard line. The historical EP value from this state is +1.06. On the play, the QB completes a 15-yard pass to their own 40-yard line. The new state (1st-and-10 from the 40) has an EP value of +1.88. The EPA of that single play is the difference: $1.88 - 1.06 = +0.82 \, EPA$.⁸⁶ A sack, turnover, or penalty would result in a large negative EPA.

**Source**: nflfastR provides this field (`epa`) for every play.⁴³ The nflfastR EP model is open-source and trained on millions of plays, using inputs like down, distance, yardline, time remaining, timeouts, and stadium type.⁴⁷

**Application**: This is the primary building block of the model. The model will not use "Yards per Game." It will use features like `Offensive_EPA_per_Play` and `Defensive_EPA_per_Play`.⁴⁸

#### 2. DVOA (Defense-adjusted Value Over Average):

**Definition**: A proprietary metric from Football Outsiders (now FTN) that measures team efficiency on a play-by-play basis, adjusted for both situation and opponent.⁴⁸

**Situational Context**: DVOA understands that a 5-yard gain on 3rd-and-4 (a highly successful play) is far more valuable than a 5-yard gain on 3rd-and-12 (an unsuccessful play).⁸⁹

**Opponent Adjustment**: Gaining 300 yards against a top-tier defense is more impressive than gaining 300 yards against a bottom-tier defense.⁸⁹ DVOA accounts for this, while raw EPA does not.

**Scale**: DVOA is expressed as a percentage relative to a league average of 0%. A +10% Offensive DVOA is 10% better than the league average.⁸⁹ For defense, negative numbers are better (e.g., a -10% Defensive DVOA is 10% better than average).

#### 3. Rolling Averages (Capturing "Form"):

**Concept**: A team's full-season average can be misleading, as it includes "stale" data from months prior. A "moving average" (or "rolling average")⁹³ provides a more accurate snapshot of a team's recent form.⁹⁴

**Application**: Instead of using a team's full-season `Offensive_EPA_per_Play`, the model should be built on a `Rolling_6_Game_Offensive_EPA_per_Play`.⁹⁵ This captures a team's current trajectory—whether they are improving or declining.

The distinction between EPA and DVOA is one of "raw impact (EPA) versus context-adjusted performance (DVOA)".⁴⁸ Since nflfastR provides raw EPA for free, a modeler can create their own "DVOA-lite" metric. This can be done by building a regression model that adjusts a team's raw EPA based on the defensive EPA of their past opponents. This "Opponent-Adjusted EPA" becomes a powerful, proprietary feature that gives the model a significant edge.

## Part III: Statistical Modeling Techniques (The Model's "Logic")

This section details the statistical "engines" that will power the model. A best-practice approach is not to build one monolithic "black box" model, but to deploy a suite of specialized models, each optimized for a specific betting market (Point Spread, Moneyline, or Total).

### Section 3.1: Foundational Tool: Team Power Ratings

The first step in any game-prediction model is to establish a baseline of team strength. A "power rating" is a single numerical value assigned to each team that represents its competitive strength relative to the rest of the league.⁹⁶ These ratings can be used to generate a "raw" point spread for any matchup.

**Formula**: $Predicted \, Spread = (Home \, Team \, Rating + HFA \, Value) - Away \, Team \, Rating$.⁶⁴

#### Method 1: The Elo Rating System

The Elo system, originally from chess¹⁰⁰, is a simple, effective, and self-correcting method. Team ratings are updated after every game based on the actual outcome versus the expected outcome.⁹⁷

**Step 1: Calculate Expected Outcome ($W_e$)**

The probability of the home team winning, based on the rating difference:

$$W_{e_{home}} = \frac{1}{1 + 10^{\frac{(Rating_{away} - Rating_{home})}{400}}}$$

**Step 2: Define Actual Outcome ($W$)**

$W = 1$ for a win, $W = 0$ for a loss, $W = 0.5$ for a tie.¹⁰³

**Step 3: Update Rating ($R_n$)**

The new rating ($R_n$) is the old rating ($R_o$) plus an adjustment:

$$R_{n_{home}} = R_{o_{home}} + K \times (W - W_{e_{home}})$$

The "K-factor" ($K$) is a constant that determines how much the rating moves after a single game.¹⁰⁴ A simple model can use a static $K=20$. A more advanced model would use a dynamic K-factor that increases based on the margin of victory, thus weighting decisive wins more heavily.¹⁰⁴

#### Method 2: Advanced Rating Systems

More complex systems, such as the Massey-Peabody (MP) ratings¹⁰⁵, move beyond simple win/loss outcomes. They are typically regression-based and built on "pure ability" metrics¹⁰⁷ or play-by-play efficiency stats, aiming to measure a team's true predictive strength, independent of luck or schedule.¹⁰⁸

An Elo model is the ideal starting point. The "Claude code" can easily implement these formulas. The resulting `elo_difference` becomes a primary predictive feature to be fed into the regression models.

### Section 3.2: Modeling for Point Spreads: Linear Regression

**Application**: Linear regression is the appropriate tool for predicting a continuous variable, such as the final point differential in a game.¹¹⁰ The model is trained on historical data to find the mathematical relationship between a set of input features and the final score margin.

**Key Features (Inputs)**: The model's feature vector X¹¹¹ should be built from the features engineered in Part II. This vector must include:

- `epa_per_play_differential` (e.g., Offense_EPA - Defense_EPA)²⁵
- `power_rating_differential` (e.g., elo_diff)⁷⁹
- `home_field_advantage` (a binary 1/0 feature)¹¹¹
- `qb_adjustment_value`⁷⁹
- `weather_adjustment_value` (from Table 3)

**The Model**: The model will solve the equation:

$$Predicted\_Spread = \beta_0 + \beta_1(epa\_diff) + \beta_2(elo\_diff) + \beta_3(is\_home\_team) + \ldots$$

The "training" process is what finds the optimal coefficients ($\beta$ values) that minimize the error between the model's prediction and the actual historical outcomes.²⁴

### Section 3.3: Modeling for Game Winners (Moneyline): Logistic Regression

**Application**: Logistic regression is the statistically correct method for predicting a binary outcome (e.g., Win = 1, Loss = 0).⁷² This is superior to simply converting a predicted point spread into a win percentage.

**The Model**: The input features (X) can be identical to those used in the linear regression model. However, the logistic model will output a probability (a value between 0.0 and 1.0) that the home team will win the game.

**Model Stacking**: A powerful technique is to stack the models. The output of the linear regression (`Predicted_Spread`) can be used as a primary input feature for the logistic regression. This allows the logistic model to find the complex, non-linear relationship between the predicted margin of victory and the actual probability of winning.¹²¹

### Section 3.4: Modeling for Totals (Over/Under): Poisson Distribution

**Application**: A Poisson distribution is a statistical tool used to model the probability of a given number of count events (e.g., goals scored, phone calls received) occurring over a fixed interval of time.¹²² This makes it a perfect tool for modeling NFL game totals, which are simply the sum of discrete scoring events (touchdowns, field goals).

**The Method**: This approach is superior to linear regression for totals because it models the underlying scoring process. The implementation follows a clear, multi-step process¹²³:

1. **Calculate League Averages**: Find the league-average points scored per game (PPG) for Home teams and Away teams over a defined period (e.g., last 3 seasons).

2. **Calculate Team "Strengths"**:
   - $Home\_Attack\_Strength = \frac{Home \, Team \, Offensive \, PPG}{League \, Avg \, Home \, PPG}$
   - $Away\_Defense\_Strength = \frac{Away \, Team \, PPG \, Allowed}{League \, Avg \, Home \, PPG}$
   - (Repeat for Away Attack and Home Defense).

3. **Calculate Expected Points ($\mu$)**:
   - $Home \, Team \, Expected \, Points \, (\mu_{home}) = Home\_Attack\_Strength \times Away\_Defense\_Strength \times League\_Avg\_Home\_PPG$
   - (Repeat for $\mu_{away}$ using Away Attack and Home Defense).

4. **Apply Poisson Formula**: With the expected point totals ($\mu_{home}$ and $\mu_{away}$), the model can calculate the probability of any exact score. The Poisson formula is:

$$P(x; \mu) = \frac{e^{-\mu} \times \mu^x}{x!}$$

The model will calculate $P(Home \, Team \, scores \, 0)$, $P(Home \, Team \, scores \, 1)$, ... $P(Home \, Team \, scores \, 50)$, and do the same for the away team.

This method allows the "Claude code" to create a probability matrix showing the discrete probability of every possible final score (e.g., 24-21: 2.1%, 27-20: 1.8%, etc.).¹²³ By summing the probabilities of all cells in this matrix where the total score is greater than the market's line (e.g., 45.5), the model can generate a precise P(Over) and P(Under), which can then be compared to the market's "true" (no-vig) probability.

## Part IV: Evaluating Model Performance and Managing Risk

A model that predicts game outcomes is a "handicapping" model. A model that generates long-term profit is a "betting" model. This final section provides the framework for backtesting, evaluation, and risk management necessary to bridge that gap.

### Section 4.1: Backtesting and Finding Positive Expected Value (+EV)

#### Backtesting:

Backtesting is the process of applying the model to historical data that it was not trained on, to simulate how it would have performed in real time.¹²⁷

**Key Method: Walk-Forward Validation**

For time-series data like sports, a simple "train-test split" is insufficient. A "walk-forward" or "rolling window" validation is the rigorous standard.⁹⁵

1. **Train**: Build (calibrate) the model using data from, for example, the 2018-2021 seasons.
2. **Test**: Use the trained model to "bet" on every game of the 2022 season, logging the model's prediction, the market line, and the hypothetical bet's outcome.
3. **Roll Forward**: Add the 2022 data to the training set.
4. **Train**: Re-calibrate the model using data from 2018-2022.
5. **Test**: Use the newly trained model to "bet" on every game of the 2023 season.

This method prevents "look-ahead bias" and accurately simulates the model's performance as it would have occurred in real-time.⁹⁵

#### Calculating Expected Value (EV):

The goal of the model is to identify +EV wagers. Expected Value is the average amount a bettor can expect to win or lose on a bet if it were placed an infinite number of times.¹³²

**Formula**:

$$EV = (Probability \, of \, Win \times Profit \, if \, Win) - (Probability \, of \, Loss \times Stake)$$

**Model Application (The "Bet" Signal)**:

- The model (from Part III) predicts $P(Cowboys -7 \, cover) = 55.0\%$.
- The market line is -110. This means:
  - Stake = $110
  - Profit if Win = $100
  - $P(Win)$ (model's) = 55.0%
  - $P(Loss)$ (model's) = 45.0%

$$EV = (0.55 \times \$100) - (0.45 \times \$110)$$
$$EV = \$55.00 - \$49.50 = +\$5.50$$

Because $EV > 0$, this is a positive expected value bet. The model's "Claude code" will log this as a "BET" in its backtest.

### Section 4.2: Key Performance Metrics: Measuring Your Edge

Once the backtest is complete, the model's output must be evaluated. Win-loss record is not the best metric. The following three KPIs are the professional standard.

#### 1. Win Rate Against the Spread (ATS):

**The Break-Even Point**: As established in Part I, the -110 line ($IP = 52.38\%$) means a bettor must win 52.38% (or 52.4%) of their ATS bets just to break even.²⁴

**Metric**: This is the model's minimum success threshold. A model that wins 51% of its bets is a losing model.

#### 2. Return on Investment (ROI):

**Formula**:

$$ROI = \left( \frac{Total \, Profit}{Total \, Amount \, Wagered} \right) \times 100$$

**Significance**: ROI, not win rate, is the true measure of profitability.¹³⁴ A model that bets on +200 moneyline underdogs may have a 40% win rate but a highly positive ROI. A model that bets on -150 favorites may have a 62% win rate but a negative ROI. ROI is the ultimate financial metric of success. A long-term ROI of 3-7% is considered professional-level.¹³⁶

#### 3. Closing Line Value (CLV):

**Definition**: CLV is the measure of the price a bettor got versus the final "closing line" (the line right before kickoff).¹³⁷

**Example**: The model bets Cowboys -3 on Tuesday. By Sunday, the line closes at Cowboys -4. The model "beat the closing line" by 1 point, achieving positive CLV. If the line closed at -2.5, the model had negative CLV.

**Why It Matters**: The closing line is the most efficient price, as it has absorbed all game news, weather reports, and, most importantly, all sharp action.¹³⁹

**The Professional's Metric**: Consistently beating the closing line is the single greatest indicator of a long-term winning bettor.¹⁴³ The actual outcome of a single game can be random. Beating the market's most efficient price over a large sample is skill.

**CLV > Outcome**: A model that bets -3 when the line closes -4 has made a good, +EV bet, even if the team fails to cover.¹⁴⁷ Conversely, a model that bets -4 when the line closes -3 has made a bad, -EV bet, even if the team wins by 10. The backtest must track CLV. A model that produces a positive ROI but negative CLV is lucky and will fail long-term. A model with a negative ROI but positive CLV is skilled but unlucky and should be trusted.¹⁴⁷

### Section 4.3: Advanced Risk and Variance Analysis

A profitable model (a 5% ROI) can still go bankrupt if risk is not managed.

#### 1. Monte Carlo Simulation:

**Purpose**: To test the model's variance and Risk of Ruin.¹⁴⁸

**Application**: The modeler takes the backtested performance metrics (e.g., 55% win rate at -110 odds) and runs 10,000 simulated seasons of 500 bets each.¹⁴⁸

**Key Output: "Risk of Ruin" (RoR)**:¹⁴⁸ The simulation will reveal the range of outcomes. It might show that even with a 55% edge, there is a 3% probability of losing the entire bankroll due to a statistically normal, but severe, losing streak. This simulation is essential for stress-testing a bankroll management strategy.

#### 2. Bankroll Management: The Kelly Criterion

**Concept**: The Kelly Criterion is a mathematical formula that calculates the optimal percentage of a bankroll to wager on a single +EV bet to maximize long-term bankroll growth.¹⁵¹

**Formula**:

$$f^* = \frac{(b \times p) - q}{b}$$

Where:
- $f^*$ = The optimal fraction of the bankroll to wager.
- $p$ = The model's "true" win probability.
- $q$ = The model's "true" loss probability (1 - p).
- $b$ = The net odds received (decimal odds - 1). (e.g., +120 odds, $b = 1.2$. -110 odds, $b = 100/110 = 0.909$).

**Example**: The model (from 4.1) sees a 55% win prob ($p=0.55$) on a -110 bet ($b=0.909$).

$$f^* = \frac{((0.909 \times 0.55) - 0.45)}{0.909}$$
$$f^* = \frac{(0.50 - 0.45)}{0.909} = \frac{0.05}{0.909} = 0.055$$

The "Full Kelly" recommendation is to bet 5.5% of the current bankroll.

#### 3. The Fractional Kelly Modification:

**The Problem**: The Full Kelly formula is notoriously volatile and, critically, it assumes the model's win probability ($p$) is perfect.¹⁵⁸ In reality, the model's edge is an estimate, and it is almost certainly overestimated.¹⁶⁰

**The Solution: "Fractional Kelly"**: The professional standard is to bet a fraction (e.g., one-quarter, one-third, or one-half) of the Full Kelly recommendation.¹⁵⁴

**Example**: For the 5.5% bet above, a "Quarter Kelly" (0.25) wager would be $5.5\% \times 0.25 = \textbf{1.375\%}$ of the bankroll. This strategy dramatically reduces volatility and Risk of Ruin, sacrificing some theoretical growth for long-term preservation.¹⁵⁸

This final step synthesizes the entire report into an actionable, automated strategy. The model's "Claude code" will execute this full workflow:

1. **Run Models** (Part II, III) to generate a "True Win %" (e.g., 55%).
2. **Scan Market** (Part I) to find the "No-Vig" probability (e.g., 50.0%).
3. **Identify Edge** (Part IV) by confirming $55\% > 50.0\%$.
4. **Calculate Wager** (Part IV) using the Fractional Kelly formula (e.g., 1.375% of bankroll).

This connects the probabilistic prediction to a disciplined, risk-managed execution, forming the complete architecture of a professional-grade betting model.

#### Table 4: Kelly Criterion Sizing Strategy (Example: "Quarter Kelly" at 0.25)

| Model's "True" Win % (p) | Market Odds | Market "True" Prob. | Model's "Edge" | Full Kelly Bet (f*) | "Quarter Kelly" Bet (f × 0.25) |
|--------------------------|-------------|---------------------|----------------|---------------------|--------------------------------|
| 53.0% | -110 ($b=0.909$) | 50.0% | 3.0% | 6.6% | 1.65% |
| 55.0% | -110 ($b=0.909$) | 50.0% | 5.0% | 11.0% | 2.75% |
| 60.0% | -110 ($b=0.909$) | 50.0% | 10.0% | 22.0% | 5.50% |
| 45.0% | +120 ($b=1.2$) | 45.5% | -0.5% | -1.0% (No Bet) | 0% (No Bet) |
| 50.0% | +120 ($b=1.2$) | 45.5% | 4.5% | 3.8% | 0.95% |

---

**End of Document**
