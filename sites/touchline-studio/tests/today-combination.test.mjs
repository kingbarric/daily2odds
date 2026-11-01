import assert from 'node:assert/strict';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
const dir=mkdtempSync(join(tmpdir(),'daily-today-'));
try{
 execFileSync('./node_modules/.bin/tsc',['lib/today-combination.ts','--module','commonjs','--target','es2022','--skipLibCheck','--outDir',dir]);
 const {todayCombination}=createRequire(import.meta.url)(join(dir,'today-combination.js'));
 const now=Date.parse('2026-11-01T12:00:00Z');
 const fixture=(id,odds,book='same')=>({id,date:'2026-11-01',league:'E0',home:id,away:'Away',asOf:'2026-11-01T09:00:00Z',kickoff:'2026-11-01T18:00:00Z',probabilities:{'1x2:home':.8},prices:{'1x2:home':{odds,bookmaker:book,sourceUpdatedAt:'2026-11-01T11:55:00Z',capturedAt:'2026-11-01T11:58:00Z',kickoff:'2026-11-01T18:00:00Z'}}});
 const a=fixture('A',1.4),b=fixture('B',1.43);
 assert.equal(todayCombination([a,b],now).odds,2.002);
 assert.equal(todayCombination([a,fixture('B',1.43,'different')],now),null);
 assert.equal(todayCombination([a,{...b,kickoff:'2026-11-01T10:00:00Z'}],now),null);
 assert.equal(todayCombination([a,{...b,date:'2026-11-02'}],now),null);
 assert.equal(todayCombination([a,b],now+3600000),null);
 assert.equal(todayCombination([a,{...b,id:'A'}],now),null);
 console.log('Today spotlight: target odds, same bookmaker, distinct matches, future kickoff and quote freshness passed');
}finally{rmSync(dir,{recursive:true,force:true})}
