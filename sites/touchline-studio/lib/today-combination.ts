import {rankForecasts,type Match} from './forecast-view';
import {searchCombinations,type Combination} from './combination-search';
/** Compare bounded searches for 2–4 distinct matches at one bookmaker. */
export function todayCombination(matches:Match[],now=Date.now()):Combination|null{
 const today=new Intl.DateTimeFormat('en-CA',{timeZone:'Europe/London',year:'numeric',month:'2-digit',day:'2-digit'}).format(now);
 const eligible=matches.filter(m=>!!m.kickoff&&Date.parse(m.kickoff)>now);
 const rows=rankForecasts(eligible,{league:'all',market:'all',start:today,end:today,minimum:0,maximum:10000,onePerMatch:false,pricedOnly:true,availableOnly:true,upcomingOnly:true,query:''},now);
 const books=[...new Set(rows.map(r=>r.price!.bookmaker))];let best:Combination|null=null;
 for(const book of books){const candidates=rows.filter(r=>r.price!.bookmaker===book);for(let count=2;count<=4;count++){const combo=searchCombinations(candidates,count,2)[0];if(combo&&(!best||combo.probability>best.probability))best=combo}}
 return best;
}
export function todayMarketLabel(key:string,home:string,away:string){const [market,s]=key.split(':');if(market==='1x2')return s==='home'?home+' win':s==='away'?away+' win':'Draw';if(market==='double_chance')return ({home_draw:'Home or draw (1X)',draw_away:'Away or draw (X2)',home_away:'Either team wins (12)'} as Record<string,string>)[s];if(market==='btts')return 'Both teams score: '+(s==='yes'?'Yes':'No');return (s==='over'?'Over ':'Under ')+market.slice(3)+' goals'}
