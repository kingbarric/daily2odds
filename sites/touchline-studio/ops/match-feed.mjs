import {readFileSync,writeFileSync,renameSync,existsSync,mkdirSync} from 'node:fs';
import path from 'node:path';

export const londonDay=(now=Date.now())=>new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/London',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(now));
const normal=value=>String(value).normalize('NFD').replace(/\p{Diacritic}/gu,'').toLowerCase().trim();
const validGoals=value=>value!==null&&value!==undefined&&String(value).trim()!==''&&Number.isInteger(Number(value))&&Number(value)>=0;
export function settleMarket(key,home,away){
 const [market,selection]=key.split(':');
 if(market==='1x2')return selection===(home>away?'home':home<away?'away':'draw');
 if(market==='double_chance')return {home_draw:home>=away,draw_away:away>=home,home_away:home!==away}[selection]??null;
 if(market==='btts')return (home>0&&away>0)===(selection==='yes');
 if(/^ou_\d+\.5$/.test(market))return (home+away>Number(market.slice(3)))===(selection==='over');
 return null;
}

/** Exact competition, ordered teams and kickoff identity. Never infer a final from elapsed time. */
export function resultForEvent(event,fixture,config,checkedAt){
 const league=config.leagueIds[event?.uid?.match(/~l:(\d+)(?:~|$)/)?.[1]];
 if(league!==fixture.league||event.competitions?.length!==1)return null;
 const game=event.competitions[0];if(game.timeValid===false)return null;
 const home=game.competitors?.find(c=>c.homeAway==='home'),away=game.competitors?.find(c=>c.homeAway==='away');
 if(!home||!away)return null;
 const same=(team,expected)=>[team.displayName,team.shortDisplayName,team.name].filter(Boolean).some(name=>normal(config.aliases[name]||name)===normal(expected));
 if(!same(home.team,fixture.home)||!same(away.team,fixture.away))return null;
 const kickoff=Date.parse(event.date),expected=Date.parse(fixture.kickoff||'');
 if(!Number.isFinite(kickoff)||!Number.isFinite(expected)||kickoff!==expected)return null;
 const info=game.status||event.status||{},type=info.type||{},name=String(type.name||'');
 const base={kickoff:fixture.kickoff,checkedAt,source:'ESPN scoreboard',eventId:String(event.id)};
 if(/POSTPON|SUSPEND/.test(name))return {...base,status:/SUSPEND/.test(name)?'suspended':'postponed'};
 if(/CANCEL/.test(name))return {...base,status:'cancelled'};
 if(/ABANDON/.test(name))return {...base,status:'abandoned'};
 let goals=[home.score,away.score];
 if(type.completed===true){
  if(Date.parse(checkedAt)<kickoff)return null;
  const extra=info.period>2||/\b(AET|ET|PEN)\b|extra|penalt/i.test(name+' '+type.detail+' '+type.shortDetail);
  if(extra){
   goals=[home,away].map(team=>{const lines=team.linescores||[];const first=lines.find(l=>Number(l.period)===1)?.value,second=lines.find(l=>Number(l.period)===2)?.value;return validGoals(first)&&validGoals(second)?Number(first)+Number(second):null});
  }
  if(!goals.every(validGoals))return {...base,status:'awaiting_result',detail:'Final regulation score awaiting confirmation'};
  return {...base,status:'finished',homeScore:Number(goals[0]),awayScore:Number(goals[1]),detail:extra?'90-minute score':'Full time'};
 }
 if(type.state==='in')return {...base,status:'live',detail:type.shortDetail||'In play',...(goals.every(validGoals)?{homeScore:Number(goals[0]),awayScore:Number(goals[1])}:{})};
 if(type.state==='pre')return {...base,status:'scheduled',detail:type.shortDetail||'Scheduled'};
 return null;
}

export function matchState(fixture,result,now=Date.now()){
 if(result&&Date.parse(result.kickoff)===Date.parse(fixture.kickoff))return result.status==='scheduled'&&Date.parse(fixture.kickoff)<=now?{...result,status:'awaiting_result'}:result;
 if(fixture.rows?.some(row=>row.score))return {status:'finished',detail:'Full time'};
 const kickoff=Date.parse(fixture.kickoff||'');
 return {status:Number.isFinite(kickoff)&&kickoff>now?'scheduled':fixture.date>londonDay(now)?'scheduled':'awaiting_result'};
}

export class MatchFeed {
 constructor({config,storage,fetcher=fetch,clock=Date.now,archivedPrices={matches:{}}}){
  Object.assign(this,{config,storage,fetcher,clock,archivedPrices});this.payloads={};this.archive=new Map();this.results={};this.lastCheckedAt=null;this.lastSuccessAt=null;this.providerStatus='checking';this.polling=false;
  if(storage&&existsSync(storage)){const saved=JSON.parse(readFileSync(storage,'utf8'));this.archive=new Map((saved.matches||[]).map(m=>[m.id,m]));this.results=saved.results||{};this.lastSuccessAt=saved.lastSuccessAt||null}
 }
 update(key,payload){
  this.payloads[key]=payload;
  for(const match of key==='history'?payload.matches||[]:key==='forecasts'?payload.matches||[]:[]){
   if(match.kind?.startsWith('Retrospective')){this.archive.set(match.id,match);continue}
   if(!match.rows&&!(Date.parse(match.asOf)<Date.parse(match.kickoff)))continue;
   const entry=match.rows?match:{...match,kind:'Saved pre-match forecast',rows:Object.entries(match.probabilities).map(([market,probability])=>({market,probability,score:null,correct:null}))};
   const old=this.archive.get(match.id);
   // Keep original pre-match probabilities; enrich missing kickoff/metadata, and accept confirmed upstream scores.
   const earliest=old&&Date.parse(old.asOf)<=Date.parse(entry.asOf)?old:entry;
   const sameKickoff=!earliest.kickoff||Date.parse(earliest.kickoff)===Date.parse(entry.kickoff);
   const rows=earliest.rows.map(row=>{const settled=sameKickoff&&entry.rows.find(r=>r.market===row.market&&r.score);return settled?{...row,score:settled.score,correct:settled.correct}:row});
   this.archive.set(match.id,{...entry,...earliest,kickoff:earliest.kickoff||entry.kickoff,probabilities:earliest.probabilities||Object.fromEntries(earliest.rows.map(r=>[r.market,r.probability])),rows});
  }
  this.save();this.compose();
 }
 save(){if(!this.storage)return;mkdirSync(path.dirname(this.storage),{recursive:true});const tmp=this.storage+'.tmp';writeFileSync(tmp,JSON.stringify({matches:[...this.archive.values()],results:this.results,lastSuccessAt:this.lastSuccessAt}));renameSync(tmp,this.storage)}
 compose(){
  const now=this.clock(),today=londonDay(now);
  const meta={lastCheckedAt:this.lastCheckedAt,lastSuccessAt:this.lastSuccessAt,providerStatus:this.providerStatus,pollIntervalSeconds:120};
  const history=[...this.archive.values()].map(entry=>{
   const state=matchState(entry,this.results[entry.id],now);
   const score=state.status==='finished'&&Number.isInteger(state.homeScore)&&Number.isInteger(state.awayScore)?`${state.homeScore}–${state.awayScore}`:null;
   return {...entry,...state,rows:entry.rows.map(row=>score?{...row,score,correct:settleMarket(row.market,state.homeScore,state.awayScore)}:row)};
  }).sort((a,b)=>b.date.localeCompare(a.date)||a.id.localeCompare(b.id));
  const forecasts=this.payloads.forecasts||{matches:[]},prices=this.payloads.prices||{matches:{}};
  const fixtures=new Map((forecasts.matches||[]).map(m=>[m.id,m]));
  for(const entry of history){if(entry.date===today&&entry.probabilities&&Number.isFinite(entry.homeGoals)&&!fixtures.has(entry.id))fixtures.set(entry.id,entry)}
  const matches=[...fixtures.values()].map(match=>({...match,...matchState(match,this.results[match.id],now),prices:{...this.archivedPrices.matches?.[match.id],...match.prices,...prices.matches?.[match.id]}}));
  this.historyBody=JSON.stringify({updatedAt:this.payloads.history?.updatedAt,matches:history,results:meta});
  this.forecastsBody=JSON.stringify({...forecasts,matches,priceCoverage:prices.coverage,pricesUpdatedAt:prices.capturedAt,results:meta});
 }
 async poll(){
  if(this.polling)return;this.polling=true;
  const now=this.clock(),checkedAt=new Date(now).toISOString(),groups=new Map();
  for(const fixture of this.archive.values()){
   const kickoff=Date.parse(fixture.kickoff||'');
   if(!Number.isFinite(kickoff)||kickoff>now+15*60000||kickoff<now-7*86400000||fixture.kind?.startsWith('Retrospective'))continue;
   const result=this.results[fixture.id];if(result?.status==='finished'&&fixture.date<londonDay(now))continue;
   const slug=this.config.slugs[fixture.league];if(!slug)continue;
   const date=new Date(kickoff).toISOString().slice(0,10).replaceAll('-',''),key=slug+':'+date;
   if(!groups.has(key))groups.set(key,{slug,date,fixtures:[]});groups.get(key).fixtures.push(fixture);
  }
  let failed=0;
  try{
   const jobs=[...groups.values()];
   const work=async()=>{while(jobs.length){const group=jobs.shift();try{
    const url=`https://site.api.espn.com/apis/site/v2/sports/soccer/${group.slug}/scoreboard?dates=${group.date}&limit=1000`;
    const response=await this.fetcher(url,{signal:AbortSignal.timeout(10000)});if(!response.ok)throw Error('Scoreboard unavailable');
    const payload=await response.json();if(!Array.isArray(payload.events)||payload.events.length>=1000)throw Error('Incomplete scoreboard');
    for(const fixture of group.fixtures){const matched=payload.events.map(event=>resultForEvent(event,fixture,this.config,checkedAt)).filter(Boolean);if(matched.length!==1)continue;const result=matched[0],old=this.results[fixture.id];if(old?.status==='finished'&&result.status!=='finished')continue;this.results[fixture.id]=result}
   }catch{failed++}}};
   await Promise.all(Array.from({length:Math.min(3,jobs.length)},work));
   this.lastCheckedAt=checkedAt;this.providerStatus=failed?'delayed':'ok';if(!failed)this.lastSuccessAt=checkedAt;
   this.save();this.compose();
  }finally{this.polling=false}
 }
}
