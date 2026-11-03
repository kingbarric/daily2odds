import assert from 'node:assert/strict';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {MatchFeed,resultForEvent,settleMarket,matchState,londonDay} from '../ops/match-feed.mjs';

const config={aliases:{'Home United':'Home'},leagueIds:{700:'E0'},slugs:{E0:'eng.1'}};
let now=Date.parse('2026-11-03T22:00:00Z');
const fixture={id:'fd:E0:2026:Home|Away',league:'E0',date:'2026-11-03',home:'Home',away:'Away',kickoff:'2026-11-03T19:45:00Z',asOf:'2026-11-03T10:00:00Z',homeGoals:1.5,awayGoals:1,probabilities:{'1x2:home':.8,'ou_2.5:under':.7,'btts:yes':.6},prices:{}};
const event=()=>({id:'1',uid:'s:600~l:700~e:1',date:fixture.kickoff,competitions:[{timeValid:true,status:{period:2,type:{name:'STATUS_FULL_TIME',state:'post',completed:true}},competitors:[{homeAway:'home',score:'2',team:{displayName:'Home United'}},{homeAway:'away',score:'1',team:{displayName:'Away'}}]}]});
const read=e=>resultForEvent(e,fixture,config,new Date(now).toISOString());
assert.equal(read(event()).status,'finished');
assert.equal(read({...event(),uid:'s:600~l:3914~e:1'}),null,'never settle an event from another league');
assert.equal(read({...event(),date:'2026-11-04T19:45:00Z'}),null,'never settle another kickoff');
let e=event();e.competitions[0].competitors.reverse();assert.equal(read(e).homeScore,2,'home/away order is explicit');
e=event();e.competitions[0].competitors[0].team.displayName='Different Home';assert.equal(read(e),null);
e=event();e.competitions[0].status.type={name:'STATUS_SECOND_HALF',state:'in',completed:false};assert.equal(read(e).status,'live');
e=event();e.competitions[0].status.type.name='STATUS_POSTPONED';assert.equal(read(e).status,'postponed');
e=event();e.competitions[0].competitors[0].score='';assert.equal(read(e).status,'awaiting_result','blank scores cannot become zero');
e=event();e.competitions[0].competitors.forEach(c=>c.score='0');assert.equal(read(e).homeScore,0,'0–0 is a valid final');
e=event();e.competitions[0].status.period=4;assert.equal(read(e).status,'awaiting_result','penalties/extra time cannot settle regulation markets without regulation scores');
e.competitions[0].competitors.forEach(c=>c.linescores=[{period:1,value:1},{period:2,value:0},{period:3,value:2}]);assert.equal(read(e).homeScore,1,'only periods one and two count');
assert.equal(matchState(fixture,null,now).status,'awaiting_result','elapsed time cannot establish a final');
assert.equal(matchState(fixture,{kickoff:fixture.kickoff,status:'scheduled'},now).status,'awaiting_result');
assert.equal(settleMarket('1x2:home',2,1),true);assert.equal(settleMarket('1x2:draw',0,0),true);assert.equal(settleMarket('double_chance:draw_away',2,1),false);assert.equal(settleMarket('ou_2.5:under',2,1),false);assert.equal(settleMarket('btts:yes',0,0),false);
assert.equal(londonDay(Date.parse('2026-07-01T23:30:00Z')),'2026-07-02','same-day retention uses London, including BST');

const dir=mkdtempSync(join(tmpdir(),'daily-results-'));
try{
 let fail=false,events=[event()],calls=0;
 const options={config,storage:join(dir,'feed.json'),clock:()=>now,fetcher:async()=>{calls++;if(fail)throw Error('Offline');return {ok:true,json:async()=>({events})}}};
 const feed=new MatchFeed(options);feed.update('forecasts',{matches:[fixture]});
 await feed.poll();let history=JSON.parse(feed.historyBody);assert.equal(history.matches[0].status,'finished');assert.deepEqual(history.matches[0].rows.map(r=>r.correct),[true,false,true]);
 const future={...fixture,id:'new',home:'New',kickoff:'2026-11-04T19:45:00Z',date:'2026-11-04'};
 feed.update('forecasts',{matches:[future]});assert.equal(JSON.parse(feed.forecastsBody).matches.length,2,'today’s finished forecast remains when the next snapshot replaces it');
 assert.equal(JSON.parse(feed.historyBody).matches.length,2,'new fixtures join the archive automatically');
 feed.update('forecasts',{matches:[{...fixture,asOf:'2026-11-03T11:00:00Z',probabilities:{'1x2:home':.5}},future]});
 assert.equal(JSON.parse(feed.historyBody).matches.find(m=>m.id===fixture.id).rows[0].probability,.8,'original pre-match probability is frozen');
 fail=true;await feed.poll();history=JSON.parse(feed.historyBody);assert.equal(history.results.providerStatus,'delayed');assert.equal(history.matches.find(m=>m.id===fixture.id).rows[0].score,'2–1','provider failure retains verified scores');
 const reloaded=new MatchFeed(options);reloaded.update('forecasts',{matches:[future]});assert.equal(JSON.parse(reloaded.historyBody).matches.find(m=>m.id===fixture.id).status,'finished','results persist through restart');
 fail=false;events=[event(),event()];delete feed.results[fixture.id];await feed.poll();assert.equal(feed.results[fixture.id],undefined,'ambiguous event matches cannot settle');
 now=Date.parse('2026-11-04T00:01:00Z');feed.compose();assert.equal(JSON.parse(feed.forecastsBody).matches.some(m=>m.id===fixture.id),true,'current snapshot is retained; UI excludes previous-day games');
 feed.update('forecasts',{matches:[future]});assert.equal(JSON.parse(feed.forecastsBody).matches.some(m=>m.id===fixture.id),false,'previous-day finals stop filling the current forecast view');
 assert.ok(calls>0);
 console.log('Match feed: identity, final-only settlement, 0–0, extra time, failure retention, frozen forecasts, new fixtures, restart persistence and London rollover passed');
}finally{rmSync(dir,{recursive:true,force:true})}
