library(survey)

options(survey.lonely.psu = "adjust")

all_args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", all_args[grepl("^--file=", all_args)])
root <- if (length(file_arg)) {
  normalizePath(file.path(dirname(file_arg[1]), ".."))
} else {
  normalizePath(".")
}
input <- file.path(root, "outputs", "derived", "full_survey_domain.csv")
output_dir <- file.path(root, "outputs", "aggregate")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

d <- read.csv(input)
for (v in c("adult", "class_l989_age", "higher_certainty_strict",
            "primary_excluding_2009_2012")) {
  d[[v]] <- d[[v]] == "True"
}

dual_indication_seqn <- c(22058, 28435, 51323)
d$full_proxy <- d$adult & d$class_l989_age
d$higher_certainty <- d$higher_certainty_strict
d$higher_certainty_no_alpha <-
  d$higher_certainty & !(d$SEQN %in% dual_indication_seqn)
d$exclude_2009_2012 <- d$primary_excluding_2009_2012
d$cycle_strata <- interaction(d$cycle, d$SDMVSTRA, drop = TRUE)
d$cycle_psu <- interaction(d$cycle, d$SDMVSTRA, d$SDMVPSU, drop = TRUE)

design <- svydesign(
  ids = ~cycle_psu,
  strata = ~cycle_strata,
  weights = ~WT20YR,
  nest = TRUE,
  data = d
)
full_df <- degf(design)

mean_ci <- function(domain, variable) {
  result <- svymean(as.formula(paste0("~", variable)), domain, na.rm = TRUE)
  interval <- confint(result)[1, ]
  c(estimate = unname(coef(result)[1]), lower = interval[1], upper = interval[2])
}

prop_ci <- function(domain, condition) {
  result <- svyciprop(
    as.formula(paste0("~I(", condition, ")")), domain,
    method = "logit", df = full_df, na.rm = TRUE
  )
  interval <- confint(result)[1, ]
  c(estimate = unname(coef(result)[1]), lower = interval[1], upper = interval[2])
}

summarize_cohort <- function(flag, label) {
  domain <- design[d[[flag]], ]
  raw <- d[d[[flag]], ]
  definitions <- list(
    age = c("mean", "RIDAGEYR", sum(!is.na(raw$RIDAGEYR))),
    non_hispanic_white = c("prop", "RIDRETH1 == 3", sum(!is.na(raw$RIDRETH1))),
    non_hispanic_black = c("prop", "RIDRETH1 == 4", sum(!is.na(raw$RIDRETH1))),
    mexican_american = c("prop", "RIDRETH1 == 1", sum(!is.na(raw$RIDRETH1))),
    other_hispanic = c("prop", "RIDRETH1 == 2", sum(!is.na(raw$RIDRETH1))),
    other_or_multiracial = c("prop", "RIDRETH1 == 5", sum(!is.na(raw$RIDRETH1))),
    poverty_income_ratio = c("mean", "INDFMPIR", sum(!is.na(raw$INDFMPIR))),
    college_graduate = c("prop", "college_grad == 1", sum(!is.na(raw$college_grad))),
    bmi = c("mean", "BMXBMI", sum(!is.na(raw$BMXBMI))),
    ever_smoked = c("prop", "ever_smoked == 1", sum(!is.na(raw$ever_smoked)))
  )
  do.call(rbind, lapply(names(definitions), function(characteristic) {
    item <- definitions[[characteristic]]
    values <- if (item[1] == "mean") mean_ci(domain, item[2]) else prop_ci(domain, item[2])
    data.frame(
      cohort = label, total_n = nrow(raw), characteristic = characteristic,
      analytic_n = as.integer(item[3]), estimate = values[1],
      lower = values[2], upper = values[3], row.names = NULL
    )
  }))
}

results <- rbind(
  summarize_cohort("full_proxy", "Full proxy"),
  summarize_cohort("higher_certainty", "Higher-certainty proxy"),
  summarize_cohort("higher_certainty_no_alpha", "Higher-certainty proxy excluding dual-indication alpha-blocker users"),
  summarize_cohort("exclude_2009_2012", "Full proxy excluding 2009-2012")
)
write.csv(results, file.path(output_dir, "survey_weighted_characteristics.csv"), row.names = FALSE)
print(results)
