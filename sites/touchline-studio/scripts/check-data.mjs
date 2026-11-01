import assert from 'node:assert/strict';
import fs from 'node:fs';
const dataset=JSON.parse(fs.readFileSync('data/forecasts.json','utf8'));
assert.equal(dataset.status,'research_snapshot');assert.equal(dataset.matches.length,104);
assert.equal(new Set(dataset.matches.map(m=>m.id)).size,dataset.matches.length);
for(const m of dataset.matches){assert.ok(m.home&&m.away&&m.date&&m.asOf);assert.ok(Number.isFinite(m.homeGoals)&&m.homeGoals>=0);assert.ok(Number.isFinite(m.awayGoals)&&m.awayGoals>=0);assert.equal(Object.keys(m.probabilities).length,14);for(const p of Object.values(m.probabilities))assert.ok(p>=0&&p<=1);assert.ok(Math.abs(['home','draw','away'].reduce((s,k)=>s+m.probabilities['1x2:'+k],0)-1)<.0001);for(const price of Object.values(m.prices))assert.ok(price.odds>1&&Number.isFinite(price.odds));}
assert.ok(!JSON.stringify(dataset).includes('apiKey'));
console.log('PASS: 104 unique matches, 14 supported selection probabilities each, valid goal rates, prices, timestamps and 1X2 probability sums.');
