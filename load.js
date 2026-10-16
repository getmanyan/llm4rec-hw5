// analysis/load.js — runs the page's own data.js and script.js in Node.
//
// The browser page and this loader execute the same two files, so every number
// printed by evaluate.js comes from the code that GitHub Pages serves.
// TensorFlow.js runs on its pure-JavaScript "cpu" backend here.
"use strict";

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const PAGE_DIR = path.join(__dirname, "..");

function requireTf() {
  try {
    return require("@tensorflow/tfjs");
  } catch (error) {
    console.error("Install TensorFlow.js first:  npm install @tensorflow/tfjs@4.22.0");
    throw error;
  }
}

async function loadApp() {
  const tf = requireTf();
  await tf.setBackend("cpu");
  await tf.ready();

  const fakeFetch = async (name) => {
    const file = path.join(PAGE_DIR, String(name));
    const text = fs.readFileSync(file, "utf8");
    return { ok: true, status: 200, text: async () => text };
  };
  const fakeDocument = { getElementById: () => null, createElement: () => ({}), body: null };

  const context = vm.createContext({
    tf,
    console,
    fetch: fakeFetch,
    document: fakeDocument,
    window: {},
    Math, Number, Array, Set, Map, Object, String, Error, JSON, Infinity, NaN,
    Int32Array, Float32Array, ArrayBuffer, Promise,
  });

  for (const file of ["data.js", "script.js"]) {
    const code = fs.readFileSync(path.join(PAGE_DIR, file), "utf8");
    vm.runInContext(code, context, { filename: file });
  }

  // Same steps as window.onload in script.js, minus the DOM.
  await vm.runInContext("loadData()", context);
  vm.runInContext(
    "(() => { const s = splitByTimestamp(); trainSet = s.train; valSet = s.val; testSet = s.test; })()",
    context,
  );

  return {
    tf,
    run: (expression) => vm.runInContext(expression, context),
    context,
  };
}

module.exports = { loadApp };
