# Extract one source zone's official PTHA18 logic tree and geometry scalars out
# of compute_rates_all_sources_session.RData.
#
# Called by fetch_official_inputs.py; can also be run standalone:
#   Rscript official_ptha_data/extract_official_tree.R <session.RData> <outdir> <zone> [<zone> ...]
#
# Only the saved session is read; nothing is recomputed. The session is the
# workspace Geoscience Australia saved when producing PTHA18, so whatever it
# holds IS the published answer.
#
# For each zone this writes, alongside the unsegmented tree, every segmented
# variant the session carries (e.g. kermadectonga2_tonga). Segments cost nothing
# extra to extract and are needed by any later segmented comparison.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
    stop("usage: extract_official_tree.R <session.RData> <outdir> <zone> [<zone> ...]")
}
SESSION <- args[1]
OUTDIR  <- args[2]
ZONES   <- args[-(1:2)]

dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

# gzfile, not a bare path: load() fails with "error reading from connection" on
# a 1.34 GB gzipped RData.
cat("loading session (1.34 GB, slow)...\n")
con <- gzfile(SESSION, "rb")
crs <- new.env()
load(con, envir = crs)
close(con)

all_names <- names(crs$source_envs)
writeLines(all_names, file.path(OUTDIR, "all_source_env_names.txt"))
cat("session holds", length(all_names), "source environments\n")

unknown <- setdiff(ZONES, all_names)
if (length(unknown) > 0) {
    stop(paste0("not in the session: ", paste(unknown, collapse = ", "),
                "\nsee ", file.path(OUTDIR, "all_source_env_names.txt")))
}

# A zone can be present in the session and still carry no logic tree.
#
# PTHA18 retires a source zone by giving it row_weight = 0 rather than deleting
# it, which is how `newhebrides` was superseded by `newhebrides2` (and likewise
# kermadectonga, puysegur, solomon, sunda, makran, timor, flores, newguinea,
# macquarienorth). compute_rates_all_sources.R then takes a shortcut branch,
# source_rate_environment_fun_row_weight_zero(), which deliberately never builds
# mw_rate_function because every event on the zone has rate zero and the full
# computation would be wasted.
#
# So the environment exists, but the function this script calls does not.
# Detected up front, because the alternative is a confusing "attempt to apply a
# non-function" several minutes into a 1.34 GB load.
has_tree <- function(nm) is.function(crs$source_envs[[nm]]$mw_rate_function)

no_tree <- ZONES[!vapply(ZONES, has_tree, logical(1))]
if (length(no_tree) > 0) {
    stop(paste0(
        "these zones carry no logic tree in the session: ",
        paste(no_tree, collapse = ", "),
        "\n\nThey are almost certainly retired zones: PTHA18 sets row_weight = 0",
        "\nin sourcezone_parameters.csv to supersede a zone rather than removing",
        "\nit, and then skips building mw_rate_function for it entirely.",
        "\nEvery event on such a zone has rate zero, so there is no published",
        "\nlogic tree to compare against.",
        "\n\nCheck the zone's row_weight, and use the replacement zone instead",
        "\n(newhebrides -> newhebrides2, sunda -> sunda2, and so on)."))
}

# Every entry that is the zone itself or the zone plus a _segment suffix.
matching <- function(zone) {
    all_names[all_names == zone | startsWith(all_names, paste0(zone, "_"))]
}

# sourcepar mixes scalars (area, mean_dip, slip) with vectors (the coupling, b
# and Mw_max axes). Scalars go to a flat key/value table; vectors are written
# separately so nothing is silently truncated to its first element.
scalar_rows <- list()
vector_rows <- list()
gcmt_rows <- list()   # LEVEL 3: the observed seismicity the update was run on

# The GCMT data LEVEL 3 used is not in sourcepar: it was passed as arguments to
# rate_of_earthquakes_greater_than_Mw_function and so lives in the closure of
# the rate function. environment() is what reaches it. Reading it here is what
# lets a re-run reproduce the posterior exactly rather than approximately.
gcmt_from_env <- function(nm, env) {
    e <- try(environment(env$mw_rate_function), silent = TRUE)
    if (inherits(e, "try-error") || is.null(e)) return(NULL)
    got <- function(x) {
        v <- try(get(x, envir = e, inherits = FALSE), silent = TRUE)
        if (inherits(v, "try-error")) NULL else v
    }
    mcd <- got("Mw_count_duration")
    if (is.null(mcd) || length(mcd) < 3) return(NULL)
    obs <- got("Mw_obs_data")
    mws <- if (!is.null(obs) && !is.null(obs$Mw)) sort(as.numeric(obs$Mw)) else numeric(0)
    # A zone with zero observed events is a real, usable answer (the update
    # still informs the weights through the expected count), so it is recorded
    # with an empty magnitude list rather than skipped.
    if (length(mws) == 0) {
        return(data.frame(source = nm, threshold_Mw = as.numeric(mcd[1]),
                          count = as.numeric(mcd[2]),
                          duration_years = as.numeric(mcd[3]),
                          index = NA_integer_, Mw = NA_real_))
    }
    data.frame(source = nm, threshold_Mw = as.numeric(mcd[1]),
               count = as.numeric(mcd[2]), duration_years = as.numeric(mcd[3]),
               index = seq_along(mws), Mw = mws)
}

for (zone in ZONES) {
    for (nm in matching(zone)) {
        cat("---", nm, "\n")
        env <- crs$source_envs[[nm]]

        g <- gcmt_from_env(nm, env)
        if (!is.null(g)) {
            gcmt_rows[[length(gcmt_rows) + 1]] <- g
            cat("   LEVEL 3 data:", g$count[1], "events above Mw",
                g$threshold_Mw[1], "in", round(g$duration_years[1], 4), "yr\n")
        } else {
            cat("   no LEVEL 3 data in the closure\n")
        }

        br <- try(env$mw_rate_function(NA, return_all_logic_tree_branches = TRUE),
                  silent = TRUE)
        if (inherits(br, "try-error")) {
            cat("   branch extraction FAILED:", as.character(br), "\n")
        } else {
            df <- br$all_par
            df$prior_prob     <- br$all_par_prob_prior
            df$posterior_prob <- br$all_par_prob
            f <- file.path(OUTDIR, paste0("logic_tree_branches_", nm, "_OFFICIAL.csv"))
            write.csv(df, f, row.names = FALSE)
            cat("   branches:", nrow(df), "->", basename(f), "\n")
        }

        sp <- env$sourcepar
        if (is.null(sp)) { cat("   no sourcepar\n"); next }
        for (k in names(sp)) {
            v <- sp[[k]]
            if (is.numeric(v) && length(v) == 1) {
                scalar_rows[[length(scalar_rows) + 1]] <-
                    data.frame(source = nm, item = k, value = as.numeric(v),
                               chr = NA_character_)
            } else if (is.numeric(v) && length(v) > 1) {
                vector_rows[[length(vector_rows) + 1]] <-
                    data.frame(source = nm, item = k, index = seq_along(v),
                               value = as.numeric(v))
            } else if (is.character(v) && length(v) == 1) {
                scalar_rows[[length(scalar_rows) + 1]] <-
                    data.frame(source = nm, item = k, value = NA_real_, chr = v)
            }
        }
    }
}

# Append rather than overwrite, so extracting a second zone later does not
# discard the first one's rows.
merge_write <- function(new, path, keys) {
    # do.call(rbind, list()) is NULL, so a run that extracted nothing would fail
    # here with "second argument must be a list" rather than saying so.
    if (is.null(new) || nrow(new) == 0) {
        cat("   nothing to write to", basename(path), "\n")
        return(0)
    }
    if (file.exists(path)) {
        old <- read.csv(path, stringsAsFactors = FALSE)
        old <- old[!(do.call(paste, old[keys]) %in% do.call(paste, new[keys])), ,
                   drop = FALSE]
        new <- rbind(old, new)
    }
    write.csv(new, path, row.names = FALSE)
    nrow(new)
}

ns <- merge_write(do.call(rbind, scalar_rows),
                  file.path(OUTDIR, "official_sourcepar_scalars.csv"),
                  c("source", "item"))
nv <- merge_write(do.call(rbind, vector_rows),
                  file.path(OUTDIR, "official_sourcepar_vectors.csv"),
                  c("source", "item", "index"))

cat("\nsourcepar rows on file: scalars", ns, " vectors", nv, "\n")

if (length(gcmt_rows) > 0) {
    ng <- merge_write(do.call(rbind, gcmt_rows),
                      file.path(OUTDIR, "official_gcmt_observations.csv"),
                      c("source", "index"))
    cat("LEVEL 3 observation rows on file:", ng, "\n")
}
cat("done.\n")
