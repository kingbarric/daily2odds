import assert from 'node:assert/strict';
import {rankForecasts} from '../lib/forecast-view.ts';
const fixture=(id,date,kickoff)=>({id,date,kickoff,home:id,away:'Away',probabilities:{'1x2:home':.8},prices:{'1x2:home':{odds:1.5,bookmaker:'test'}},asOf:'2026-11-01T00:00:00Z',league:'E0'});
const matches=[fixture('finished','2026-11-01','2026-11-01T09:00:00Z'),fixture('today','2026-11-01','2026-11-01T18:00:00Z'),fixture('tomorrow','2026-11-02','2026-11-02T18:00:00Z')];
const filters={league:'all',market:'all',start:'2026-11-01',end:'2026-11-01',minimum:0,maximum:10000,onePerMatch:false,pricedOnly:false,availableOnly:true,upcomingOnly:true,query:''};
assert.deepEqual(rankForecasts(matches,filters,Date.parse('2026-11-01T12:00:00Z')).map(r=>r.match.id),['today']);
assert.deepEqual(rankForecasts(matches,{...filters,end:'2026-11-02'},Date.parse('2026-11-01T12:00:00Z')).map(r=>r.match.id),['today','tomorrow']);
console.log('Date range and pre-kickoff combination filters passed');
