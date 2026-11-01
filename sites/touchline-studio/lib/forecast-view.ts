export type Price={odds:number;bookmaker:string;capturedAt?:string;sourceUpdatedAt?:string;kickoff?:string};
export type Match={id:string;date:string;league:string;home:string;away:string;asOf:string;kickoff?:string;modelSource?:string;teamNewsStatus?:string;teamContext?:Record<string,{team:string;historicalMatches:number;recent:{date:string;opponent:string;scored:number;conceded:number;result:string}[]}>;homeGoals:number;awayGoals:number;probabilities:Record<string,number>;prices:Record<string,Price>};
export type ForecastRow={id:string;match:Match;market:string;selection:string;probability:number;price?:Price;priceStatus:'fresh'|'stale'|'unavailable';likelihood:string};
export type ViewFilters={league:string;market:string;start:string;end:string;minimum:number;maximum:number;onePerMatch:boolean;pricedOnly:boolean;availableOnly?:boolean;upcomingOnly?:boolean;query:string};
export function likelihood(p:number){return p>=.8?'higher likelihood':p>=.6?'moderate likelihood':'lower likelihood'}
export function rankForecasts(matches:Match[],filters:ViewFilters,now=Date.now()):ForecastRow[]{
 let rows:ForecastRow[]=[];
 for(const match of matches){
  if(filters.upcomingOnly&&match.kickoff&&Date.parse(match.kickoff)<=now)continue;
  if(filters.league!=='all'&&match.league!==filters.league)continue;
  if(match.date<filters.start||match.date>filters.end)continue;
  if(!`${match.home} ${match.away}`.toLowerCase().includes(filters.query.toLowerCase()))continue;
  for(const [key,p] of Object.entries(match.probabilities)){
   const [market,selection]=key.split(':');if(filters.market!=='all'&&market!==filters.market)continue;
   if(!Number.isFinite(p)||p<filters.minimum/100)continue;
   const price=match.prices[key];const source=Date.parse(price?.sourceUpdatedAt||'');const captured=Date.parse(price?.capturedAt||'');const kickoff=Date.parse(price?.kickoff||'');
   const usable=!!price&&Number.isFinite(price.odds)&&price.odds>1;
   const fresh=usable&&source<=captured&&captured<=now&&now-source<=3600000&&now<kickoff;
   if(filters.availableOnly&&!usable)continue;
   if(filters.pricedOnly&&!fresh)continue;
   rows.push({id:match.id+'::'+key,match,market,selection,probability:p,price:usable?price:undefined,priceStatus:fresh?'fresh':usable?'stale':'unavailable',likelihood:likelihood(p)});
  }
 }
 rows.sort((a,b)=>b.probability-a.probability||a.match.date.localeCompare(b.match.date)||a.id.localeCompare(b.id));
 if(filters.onePerMatch){const used=new Set<string>();rows=rows.filter(r=>{if(used.has(r.match.id))return false;used.add(r.match.id);return true})}
 return rows.slice(0,Math.min(10000,Math.max(1,filters.maximum)));
}

/** Arithmetic comparison of explicitly selected displayed prices; no order/ticket API. */
export function combineDisplayedPrices(rows:ForecastRow[],now=Date.now()){
 const seen=new Set<string>();let total=1;
 for(const row of rows){
  if(seen.has(row.match.id))throw new Error('Choose only one outcome from each match.');
  seen.add(row.match.id);
  if(!row.price||!Number.isFinite(row.price.odds)||row.price.odds<=1)throw new Error('Every selected outcome needs a genuine displayed decimal price.');
  total*=row.price.odds;
  if(!Number.isFinite(total))throw new Error('The combined number is too large to display.');
 }
 return {count:rows.length,combinedOdds:rows.length?total:0,bookmakers:[...new Set(rows.map(r=>r.price!.bookmaker))],staleCount:rows.filter(r=>r.priceStatus!=='fresh'||!r.price?.sourceUpdatedAt||now-Date.parse(r.price.sourceUpdatedAt)>3600000||!r.price.kickoff||now>=Date.parse(r.price.kickoff)).length};
}
