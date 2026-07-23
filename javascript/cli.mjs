#!/usr/bin/env node

import { readFileSync, writeFileSync } from "node:fs";
import {
  DEFAULT_MAX_POINTS,
  decodeCpos,
  encodePos,
  inspectCpos,
  pointsToPos,
} from "./cpos.js";

function usage() {
  console.error(`usage:
  node javascript/cli.mjs encode INPUT.pos OUTPUT.cpos [--max-points N]
  node javascript/cli.mjs decode INPUT.cpos OUTPUT.pos
  node javascript/cli.mjs inspect INPUT.cpos`);
  process.exitCode = 2;
}

const [command, inputPath, outputPath, ...rest] = process.argv.slice(2);
if (!command || !inputPath) {
  usage();
} else if (command === "encode") {
  if (!outputPath) {
    usage();
  } else {
    let maxPoints = DEFAULT_MAX_POINTS;
    for (let index = 0; index < rest.length; index += 1) {
      if (rest[index] === "--max-points" && rest[index + 1]) {
        maxPoints = Number(rest[index + 1]);
        index += 1;
      } else {
        throw new Error(`unknown argument: ${rest[index]}`);
      }
    }
    const source = readFileSync(inputPath);
    const encoded = encodePos(source, { maxPoints });
    writeFileSync(outputPath, new Uint8Array(encoded));
  }
} else if (command === "decode") {
  if (!outputPath) {
    usage();
  } else {
    const decoded = decodeCpos(readFileSync(inputPath));
    writeFileSync(outputPath, new Uint8Array(pointsToPos(decoded.points)));
  }
} else if (command === "inspect") {
  const header = inspectCpos(readFileSync(inputPath));
  console.log(JSON.stringify({
    ...header,
    containerVersion: header.containerVersion.join("."),
    algorithmVersion: header.algorithmVersion.join("."),
    payloadCrc32: header.payloadCrc32.toString(16).padStart(8, "0"),
  }, null, 2));
} else {
  usage();
}
