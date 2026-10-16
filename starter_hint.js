// analysis/starter_hint.js — what happens with the loss suggested in the
// trainMF comment of the starter (L2 on the WHOLE factor matrices + Adam).
//
//   node analysis/starter_hint.js      (~2 min)
//
// It uses the page's data and split, the same initialization (seeds 42/43) and
// batch size 64, and trains one epoch with exactly the starter's loss:
//   mse(batch) + reg * (sum of all userFactors^2 + sum of all itemFactors^2)
"use strict";

const { loadApp } = require("./load.js");

(async () => {
  const { tf, run } = await loadApp();
  const trainSet = run("trainSet");
  const valSet = run("valSet");
  const fmt = (x) => x.toFixed(4);

  for (const reg of [0.05, 0.001]) {
    run("initializeFactors(trainSet, NUM_FACTORS)");
    const U = run("model.userFactors");
    const I = run("model.itemFactors");
    const mu = run("model.globalMean");
    const opt = tf.train.adam(0.01);
    const n = trainSet.length;
    for (let start = 0; start < n; start += 64) {
      const batch = trainSet.slice(start, start + 64);
      const bu = tf.tensor1d(batch.map((r) => r.userId), "int32");
      const bi = tf.tensor1d(batch.map((r) => r.movieId), "int32");
      const y = tf.tensor1d(batch.map((r) => r.rating), "float32");
      opt.minimize(() => tf.tidy(() => {
        const u = tf.oneHot(bu, U.shape[0]).toFloat().matMul(U);
        const i = tf.oneHot(bi, I.shape[0]).toFloat().matMul(I);
        const pred = tf.sum(u.mul(i), 1).add(mu);
        const mse = tf.losses.meanSquaredError(y, pred);
        const l2 = U.square().sum().add(I.square().sum()).mul(reg);
        return mse.add(l2);
      }), true, [U, I]).dispose();
      tf.dispose([bu, bi, y]);
    }
    opt.dispose();
    const maxAbs = Math.max(...run("model.itemFactors.abs().max().dataSync()"), ...run("model.userFactors.abs().max().dataSync()"));
    console.log(
      `reg ${reg}: after 1 epoch val RMSE ${fmt(run("computeRMSEFor(valSet)"))}, ` +
      `baseline val RMSE ${fmt(run("computeBaselineRMSE(valSet)"))}, largest |factor| ${maxAbs.toExponential(2)}`,
    );
  }
  console.log("In both cases the validation RMSE equals the mean-rating baseline: the penalty reaches every");
  console.log("row at every batch, and Adam turns it into a step of about lr toward zero even when reg is small.");
})();
