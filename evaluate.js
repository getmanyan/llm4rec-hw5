// analysis/evaluate.js — every number in the HW5 report.
//
// Usage (from the repository root):
//   npm install @tensorflow/tfjs@4.22.0         (once)
//   node analysis/evaluate.js                   (everything, ~30 min on CPU)
//   node analysis/evaluate.js --skip-adam       (without the Adam comparison, ~15 min)
//
// Training is seeded, so a re-run prints the same numbers (only timings differ).
//
// It runs the page's own data.js and script.js (see load.js), with the page's
// default hyper-parameters, on the TensorFlow.js "cpu" backend.
"use strict";

const { loadApp } = require("./load.js");

const args = process.argv.slice(2);
const argValue = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] !== undefined ? Number(args[i + 1]) : fallback;
};

// Page defaults (index.html inputs).
const OPTS = {
  epochs: argValue("--epochs", 20),
  lr: 0.01,
  reg: 0.05,
  batchSize: 64,
};
const SKIP_ADAM = args.includes("--skip-adam");
// Optional override of the two users discussed in the report (Section 5).
const USER_OVERRIDE = [argValue("--user1", NaN), argValue("--user2", NaN)];
const K = 10;

const fmt = (x, d = 4) => (Number.isFinite(x) ? x.toFixed(d) : "—");
const pct = (x, d = 1) => (Number.isFinite(x) ? (100 * x).toFixed(d) + "%" : "—");
const line = (s = "") => console.log(s);
const head = (s) => line(`\n=== ${s} ===`);

(async () => {
  const app = await loadApp();
  const { tf, run } = app;
  const movies = run("movies");
  const ratings = run("ratings");
  const trainSet = run("trainSet");
  const valSet = run("valSet");
  const testSet = run("testSet");
  const numUsers = run("numUsers");
  const byId = new Map(movies.map((m) => [m.id, m]));

  // ------------------------------------------------------------------ data
  head("1. Data");
  const maxMovieId = movies.reduce((m, x) => Math.max(m, x.id), 0);
  const ratedMovies = new Set(ratings.map((r) => r.movieId)).size;
  const cells = numUsers * movies.length;
  line(`users ${numUsers}, movies ${movies.length}, ratings ${ratings.length}`);
  line(`movies with at least one rating: ${ratedMovies}`);
  line(`dense array (page): ${numUsers + 1} x ${maxMovieId + 1} (row/col 0 unused)`);
  line(`observed share of the ${numUsers} x ${movies.length} matrix: ${pct(ratings.length / cells, 2)}; sparsity ${pct(1 - ratings.length / cells, 2)}`);
  const perUser = new Map();
  for (const r of ratings) perUser.set(r.userId, (perUser.get(r.userId) || 0) + 1);
  const counts = Array.from(perUser.values()).sort((a, b) => a - b);
  line(`ratings per user: min ${counts[0]}, median ${counts[Math.floor(counts.length / 2)]}, max ${counts[counts.length - 1]}`);
  const dist = [1, 2, 3, 4, 5].map((v) => ratings.filter((r) => r.rating === v).length);
  line(`rating distribution 1..5: ${dist.join(" / ")} (share >= 4: ${pct((dist[3] + dist[4]) / ratings.length)})`);
  line("missing ratings: the model trains only on the observed (user, movie, rating) triples;");
  line("  ratingMatrix (0 = unobserved) is built by data.js but never read by script.js.");

  // ----------------------------------------------------------------- split
  head("2. Split (per user, by timestamp, 80/10/10)");
  const total = trainSet.length + valSet.length + testSet.length;
  line(`train ${trainSet.length} (${pct(trainSet.length / total)}), val ${valSet.length} (${pct(valSet.length / total)}), test ${testSet.length} (${pct(testSet.length / total)})`);
  line(`users skipped by the split (fewer than 5 ratings): ${run("SPLIT_EXCLUSIONS").length}`);
  const trainUsers = new Set(trainSet.map((r) => r.userId));
  const trainMovies = new Set(trainSet.map((r) => r.movieId));
  const evalUsers = new Set([...valSet, ...testSet].map((r) => r.userId));
  line(`users in val/test absent from train: ${[...evalUsers].filter((u) => !trainUsers.has(u)).length}`);
  line(`distinct movies: train ${trainMovies.size}, val ${new Set(valSet.map((r) => r.movieId)).size}, test ${new Set(testSet.map((r) => r.movieId)).size}`);
  for (const [name, split] of [["val", valSet], ["test", testSet]]) {
    const cold = split.filter((r) => !trainMovies.has(r.movieId));
    line(`${name} ratings on movies absent from train: ${cold.length} (${new Set(cold.map((r) => r.movieId)).size} movies); kept in RMSE, the model predicts mu + p_u.q_i with an untrained q_i`);
  }
  const timeline = (split) => {
    const ts = split.map((r) => r.timestamp);
    return [new Date(Math.min(...ts) * 1000).toISOString().slice(0, 10), new Date(Math.max(...ts) * 1000).toISOString().slice(0, 10)];
  };
  line(`date ranges: train ${timeline(trainSet).join(" .. ")}, test ${timeline(testSet).join(" .. ")}`);

  // -------------------------------------------------------------- training
  head("3. Training (page defaults)");
  line(`opts ${JSON.stringify(OPTS)}, factors 16, optimizer SGD (tf.train.sgd), seeds 42/43 (init) and 2026 (batch order), backend ${tf.getBackend()}`);
  const t0 = Date.now();
  run("initializeFactors(trainSet, NUM_FACTORS)");
  const history = run(`trainingHistory = trainMF(trainSet, valSet, ${JSON.stringify(OPTS)})`);
  const best = run("bestCheckpoint = selectBestCheckpoint(trainingHistory)");
  const seconds = (Date.now() - t0) / 1000;
  line("epoch  trainRMSE  valRMSE");
  for (const h of history) line(`${String(h.epoch).padStart(5)}  ${fmt(h.trainRMSE)}     ${fmt(h.valRMSE)}${h.epoch === best.epoch ? "  <- selected" : ""}`);
  line(`training time: ${seconds.toFixed(0)} s (${(seconds / OPTS.epochs).toFixed(1)} s per epoch)`);

  // ------------------------------------------------------------------ RMSE
  head("4. RMSE (model = selected checkpoint)");
  const mu = run("model.globalMean");
  const baseVal = run("computeBaselineRMSE(valSet)");
  const baseTest = run("computeBaselineRMSE(testSet)");
  const mfVal = run("computeRMSEFor(valSet)");
  const mfTest = run("computeRMSEFor(testSet)");
  line(`training mean mu = ${fmt(mu)}`);
  line(`validation: baseline ${fmt(baseVal)}, MF ${fmt(mfVal)} (selected epoch ${best.epoch}, history says ${fmt(best.valRMSE)})`);
  line(`test:       baseline ${fmt(baseTest)}, MF ${fmt(mfTest)}  -> ${pct(1 - mfTest / baseTest)} lower`);
  const lastEpoch = history[history.length - 1];
  line(`last epoch ${lastEpoch.epoch}: train ${fmt(lastEpoch.trainRMSE)}, val ${fmt(lastEpoch.valRMSE)}`);

  // Independent check: plain JavaScript over the factor arrays, no TF.js ops.
  const F = 16;
  const snapshot = () => ({ U: run("model.userFactors.dataSync()").slice(), Q: run("model.itemFactors.dataSync()").slice(), mu: run("model.globalMean") });
  const S = snapshot();
  const plainScore = (W, u, i) => {
    let d = 0;
    for (let f = 0; f < F; f++) d += W.U[u * F + f] * W.Q[i * F + f];
    return W.mu + d;
  };
  const plainRMSE = (W, split) => Math.sqrt(split.reduce((acc, r) => acc + (Math.min(5, Math.max(1, plainScore(W, r.userId, r.movieId))) - r.rating) ** 2, 0) / split.length);
  line(`check, plain-JS RMSE from the factor arrays: val ${fmt(plainRMSE(S, valSet))}, test ${fmt(plainRMSE(S, testSet))}`);

  // ----------------------------------------------- helpers for the lists
  const popularity = new Map();
  for (const r of trainSet) popularity.set(r.movieId, (popularity.get(r.movieId) || 0) + 1);
  const popRank = new Map(
    [...popularity.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).map(([id], i) => [id, i + 1]),
  );
  const historyOf = new Map();
  for (const r of [...trainSet, ...valSet]) {
    if (!historyOf.has(r.userId)) historyOf.set(r.userId, []);
    historyOf.get(r.userId).push(r);
  }
  const likedGenresOf = (u, n = 3) => {
    const c = new Map();
    for (const r of historyOf.get(u)) {
      if (r.rating < 4) continue;
      for (const g of byId.get(r.movieId).genres) c.set(g, (c.get(g) || 0) + 1);
    }
    return [...c.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, n);
  };
  const allUsers = [...trainUsers].sort((a, b) => a - b);

  // Summary of every user's Top-10 for the current model.
  const describeAll = (label) => {
    const lists = new Map();
    const slotCount = new Map();
    let slots = 0, top100 = 0, rare = 0, fam = 0, violations = 0, notInTrain = 0;
    for (const u of allUsers) {
      const recs = run(`recommendTopK(${u}, ${K})`);
      lists.set(u, recs);
      const seen = new Set(historyOf.get(u).map((r) => r.movieId));
      const liked = new Set(likedGenresOf(u).map(([g]) => g));
      for (const rec of recs) {
        slots += 1;
        slotCount.set(rec.movieId, (slotCount.get(rec.movieId) || 0) + 1);
        const pop = popularity.get(rec.movieId) || 0;
        if ((popRank.get(rec.movieId) || 1e9) <= 100) top100 += 1;
        if (pop <= 5) rare += 1;
        if (byId.get(rec.movieId).genres.some((g) => liked.has(g))) fam += 1;
        if (seen.has(rec.movieId)) violations += 1;
        if (!trainMovies.has(rec.movieId)) notInTrain += 1;
      }
    }
    const recPop = [...slotCount.entries()].reduce((a, [id, c]) => a + c * (popularity.get(id) || 0), 0) / slots;
    line(`[${label}] users ${allUsers.length}, slots ${slots}, recommended movie already rated in train/val: ${violations}, slots on movies absent from train: ${notInTrain}`);
    line(`[${label}] distinct movies across all Top-10 lists: ${slotCount.size} of ${trainMovies.size} train movies (${pct(slotCount.size / trainMovies.size)})`);
    line(`[${label}] slots: ${pct(top100 / slots)} from the 100 most-rated movies, ${pct(rare / slots)} from movies with <= 5 train ratings, ${pct(fam / slots)} share a top-3 liked genre of the user`);
    line(`[${label}] mean train popularity of recommended slots: ${fmt(recPop, 0)} ratings`);
    const most = [...slotCount.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).slice(0, 8);
    line(`[${label}] most recommended movies:`);
    for (const [id, c] of most) line(`    ${byId.get(id).title}: ${c} users (${pct(c / allUsers.length)}), ${popularity.get(id) || 0} train ratings (rank ${popRank.get(id)})`);
    return lists;
  };

  const showUser = (u, recs, test) => {
    const hist = historyOf.get(u);
    const likedGenres = likedGenresOf(u);
    const likedSet = new Set(likedGenres.map(([g]) => g));
    const histPop = hist.reduce((a, r) => a + (popularity.get(r.movieId) || 0), 0) / hist.length;
    const testMap = new Map(test.filter((r) => r.userId === u).map((r) => [r.movieId, r.rating]));
    line(`\nuser ${u}: ${hist.length} ratings in train+val (mean ${fmt(hist.reduce((a, r) => a + r.rating, 0) / hist.length, 2)}), ${hist.filter((r) => r.rating >= 4).length} rated >= 4, ${testMap.size} in test`);
    line(`  top genres among movies rated >= 4: ${likedGenres.map(([g, c]) => `${g} ${c}`).join(", ")}`);
    line(`  mean popularity of rated movies: ${fmt(histPop, 0)} train ratings`);
    const favs = hist.filter((r) => r.rating === 5).sort((a, b) => (popularity.get(b.movieId) || 0) - (popularity.get(a.movieId) || 0));
    line(`  5-star movies, most rated first (${favs.length}): ${favs.slice(0, 10).map((r) => byId.get(r.movieId).title).join("; ")}`);
    let fam = 0, top100 = 0, popSum = 0, inTest = 0;
    recs.forEach((rec, i) => {
      const m = byId.get(rec.movieId);
      const isFam = m.genres.some((g) => likedSet.has(g));
      if (isFam) fam += 1;
      const rank = popRank.get(rec.movieId);
      if (rank && rank <= 100) top100 += 1;
      popSum += popularity.get(rec.movieId) || 0;
      const t = testMap.has(rec.movieId) ? ` | in test, rated ${testMap.get(rec.movieId)}` : "";
      if (t) inTest += 1;
      line(`  ${String(i + 1).padStart(2)}. ${rec.title} | score ${fmt(rec.score, 3)} | ${m.genres.join("/")} | ${popularity.get(rec.movieId) || 0} train ratings (rank ${rank || "—"})${isFam ? "" : " | NEW GENRE"}${t}`);
    });
    line(`  summary: ${fam}/10 share a top-3 liked genre, ${top100}/10 in the 100 most-rated movies, mean popularity ${fmt(popSum / recs.length, 0)} train ratings, ${inTest}/10 rated by the user in test`);
  };

  // --------------------------------------------- all users, then two users
  head("5. All users: what the Top-10 lists look like (SGD, page defaults)");
  const lists = describeAll("SGD");

  // The two report users: among users with at least 100 ratings in train+val,
  // the one whose list shares a top-3 liked genre most often and the one least
  // often (ties -> lowest id). They show the two ends of "familiar vs new".
  const familiarShare = (u) => {
    const liked = new Set(likedGenresOf(u).map(([g]) => g));
    return lists.get(u).filter((rec) => byId.get(rec.movieId).genres.some((g) => liked.has(g))).length;
  };
  const heavy = allUsers.filter((u) => historyOf.get(u).length >= 100);
  const byFam = heavy.map((u) => [u, familiarShare(u)]).sort((a, b) => a[1] - b[1] || a[0] - b[0]);
  const mostFamiliar = byFam.reduce((b, x) => (x[1] > b[1] ? x : b), byFam[0])[0];
  const leastFamiliar = byFam[0][0];
  const reportUsers = [
    Number.isFinite(USER_OVERRIDE[0]) ? USER_OVERRIDE[0] : mostFamiliar,
    Number.isFinite(USER_OVERRIDE[1]) ? USER_OVERRIDE[1] : leastFamiliar,
  ];
  head("6. Top-10 for the two report users");
  line(`users with >= 100 ratings in train+val: ${heavy.length}; familiar slots per list among them: min ${byFam[0][1]}, median ${byFam[Math.floor(byFam.length / 2)][1]}, max ${byFam[byFam.length - 1][1]}`);
  line(`chosen: user ${reportUsers[0]} (most familiar list) and user ${reportUsers[1]} (least familiar list)`);
  for (const u of reportUsers) showUser(u, lists.get(u), testSet);

  // One pair by hand: the first test rating of the first report user.
  head("7. One prediction by hand");
  const pair = testSet.find((r) => r.userId === reportUsers[0]);
  const pu = Array.from(S.U.slice(pair.userId * F, pair.userId * F + F));
  const qi = Array.from(S.Q.slice(pair.movieId * F, pair.movieId * F + F));
  const dot = pu.reduce((acc, x, f) => acc + x * qi[f], 0);
  const pred = Math.min(5, Math.max(1, S.mu + dot));
  line(`user ${pair.userId}, movie ${pair.movieId} (${byId.get(pair.movieId).title}), true test rating ${pair.rating}`);
  line(`  p_u = [${pu.map((x) => x.toFixed(3)).join(", ")}]`);
  line(`  q_i = [${qi.map((x) => x.toFixed(3)).join(", ")}]`);
  line(`  p_u . q_i = ${fmt(dot)}; mu + dot = ${fmt(S.mu)} + ${fmt(dot)} = ${fmt(S.mu + dot)}; clamp to [1, 5] -> ${fmt(pred)}; predictRating() = ${fmt(run(`predictRating(${pair.userId}, ${pair.movieId})`))}`);
  line(`  squared error = (${fmt(pred)} - ${pair.rating})^2 = ${fmt((pred - pair.rating) ** 2)}`);

  // ------------------------------------------------------------- Recall@10
  head("8. Recall@10 (relevant = test rating >= 4.0, candidates = train movies)");
  const recallReport = (label) => {
    const rec = run(`recallAtK(${K})`);
    const relevantTotal = rec.perUser.reduce((a, p) => a + p.relevant, 0);
    const hits = rec.perUser.reduce((a, p) => a + p.hit, 0);
    line(`[${label}] Recall@10 = ${fmt(rec.overallRecall)} over ${rec.perUser.length} users; hits ${hits} of ${relevantTotal} relevant test movies; users with >= 1 hit: ${rec.perUser.filter((p) => p.hit > 0).length}`);
    return rec;
  };
  const rec = recallReport("SGD");
  const testUsers = new Set(testSet.map((r) => r.userId));
  const unseenRatings = testSet.filter((r) => r.rating >= 4 && !trainMovies.has(r.movieId)).length;
  line(`excluded: ${rec.excluded.unseenIds.length} relevant test movie ids absent from train (${unseenRatings} ratings): [${rec.excluded.unseenIds.join(", ")}]`);
  line(`excluded: ${rec.excluded.usersWithoutRelevant.length} of ${testUsers.size} test users have no relevant test movie among the candidates`);

  // Independent check of Recall@10 from the plain factor arrays (full sort).
  const relevantBy = new Map();
  for (const r of testSet) {
    if (r.rating >= 4 && trainMovies.has(r.movieId)) {
      if (!relevantBy.has(r.userId)) relevantBy.set(r.userId, new Set());
      relevantBy.get(r.userId).add(r.movieId);
    }
  }
  const plainRecall = (W) => {
    let sum = 0;
    for (const [u, rel] of relevantBy) {
      const seen = new Set(historyOf.get(u).map((r) => r.movieId));
      const scored = movies.filter((m) => !seen.has(m.id)).map((m) => [m.id, plainScore(W, u, m.id)]);
      scored.sort((a, b) => b[1] - a[1] || a[0] - b[0]);
      sum += scored.slice(0, K).filter(([id]) => rel.has(id)).length / rel.size;
    }
    return sum / relevantBy.size;
  };
  line(`check, plain-JS Recall@10 (sort of all scores, no tf.topk): ${fmt(plainRecall(S))} over ${relevantBy.size} users`);

  // Simple reference points.
  const byPop = [...popularity.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0]).map(([id]) => id);
  let popR = 0;
  for (const [u, rel] of relevantBy) {
    const seen = new Set(historyOf.get(u).map((r) => r.movieId));
    const top = byPop.filter((id) => !seen.has(id)).slice(0, K);
    popR += top.filter((id) => rel.has(id)).length / rel.size;
  }
  line(`popularity list Recall@10 (10 most-rated train movies the user has not rated): ${fmt(popR / relevantBy.size)}`);
  const meanSeen = allUsers.reduce((a, u) => a + historyOf.get(u).length, 0) / allUsers.length;
  line(`random list (expected): about ${fmt(K / (movies.length - meanSeen))} (10 of ~${fmt(movies.length - meanSeen, 0)} unrated movies)`);

  // ----------------------------------------------------- Adam comparison
  if (!SKIP_ADAM) {
    head("9. Same model trained with Adam (optimizer option of trainMF)");
    const adamOpts = { ...OPTS, optimizer: "adam" };
    line(`opts ${JSON.stringify(adamOpts)}`);
    run("initializeFactors(trainSet, NUM_FACTORS)");
    const adamHistory = run(`trainMF(trainSet, valSet, ${JSON.stringify(adamOpts)})`);
    const adamBest = run("selectBestCheckpoint(trainingHistory)");
    line("epoch  trainRMSE  valRMSE");
    for (const h of adamHistory) line(`${String(h.epoch).padStart(5)}  ${fmt(h.trainRMSE)}     ${fmt(h.valRMSE)}${h.epoch === adamBest.epoch ? "  <- selected" : ""}`);
    line(`[Adam] test RMSE ${fmt(run("computeRMSEFor(testSet)"))} (baseline ${fmt(baseTest)}, SGD ${fmt(mfTest)})`);
    const adamLists = describeAll("Adam");
    recallReport("Adam");
    for (const u of reportUsers) {
      line(`\n[Adam] user ${u} Top-10:`);
      adamLists.get(u).forEach((r, i) => line(`  ${String(i + 1).padStart(2)}. ${r.title} | score ${fmt(r.score, 3)} | ${popularity.get(r.movieId) || 0} train ratings`));
    }
    // Why: vector length of movies by how often they were rated.
    const W = snapshot();
    const norm = (id) => Math.sqrt(Array.from({ length: F }, (_, f) => W.Q[id * F + f] ** 2).reduce((a, b) => a + b, 0));
    const normS = (id) => Math.sqrt(Array.from({ length: F }, (_, f) => S.Q[id * F + f] ** 2).reduce((a, b) => a + b, 0));
    const groups = [["1-5 ratings", 1, 5], ["6-50", 6, 50], ["51-200", 51, 200], [">200", 201, 1e9]];
    line("\nmean length of movie vectors |q_i| by number of train ratings (SGD vs Adam):");
    for (const [name, lo, hi] of groups) {
      const ids = [...popularity.entries()].filter(([, c]) => c >= lo && c <= hi).map(([id]) => id);
      const m = (f) => ids.reduce((a, id) => a + f(id), 0) / ids.length;
      line(`  ${name.padEnd(12)} ${String(ids.length).padStart(4)} movies: SGD ${fmt(m(normS), 3)}, Adam ${fmt(m(norm), 3)}`);
    }
  }

  // ------------------------------------------------- gather gradient timing
  head("10. Why training uses one-hot matMul instead of tf.gather");
  const timeSteps = (useGather, steps = 20) => {
    const Uv = tf.variable(tf.randomNormal([944, 16], 0, 0.05, "float32", 1));
    const Iv = tf.variable(tf.randomNormal([1683, 16], 0, 0.05, "float32", 2));
    const opt = tf.train.sgd(0.01);
    const bu = new Int32Array(64).map((_, k) => 1 + ((k * 37) % 943));
    const bi = new Int32Array(64).map((_, k) => 1 + ((k * 101) % 1682));
    const y = new Float32Array(64).fill(4);
    const start = Date.now();
    for (let s = 0; s < steps; s++) {
      opt.minimize(() => tf.tidy(() => {
        const ub = tf.tensor1d(bu, "int32");
        const ib = tf.tensor1d(bi, "int32");
        const u = useGather ? tf.gather(Uv, ub) : tf.oneHot(ub, 944).toFloat().matMul(Uv);
        const i = useGather ? tf.gather(Iv, ib) : tf.oneHot(ib, 1683).toFloat().matMul(Iv);
        return tf.losses.meanSquaredError(tf.tensor1d(y), tf.sum(u.mul(i), 1).add(3.5));
      }), true, [Uv, Iv]).dispose();
    }
    const ms = (Date.now() - start) / steps;
    Uv.dispose(); Iv.dispose(); opt.dispose();
    return ms;
  };
  const gatherMs = timeSteps(true);
  const oneHotMs = timeSteps(false);
  line(`one batch of 64 on backend ${tf.getBackend()}: tf.gather ${gatherMs.toFixed(0)} ms, one-hot matMul ${oneHotMs.toFixed(0)} ms (${(gatherMs / oneHotMs).toFixed(1)}x)`);
  line("both select the same rows; the difference is the gradient kernel (UnsortedSegmentSum vs MatMul).");
  line("\ndone");
})();
