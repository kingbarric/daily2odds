import type {ForecastRow} from './forecast-view';
export type Combination={rows:ForecastRow[];odds:number;probability:number};
/** Bounded log-price beam search. Approximate ranking, not a global optimum guarantee. */
export function searchCombinations(rows:ForecastRow[],count:number,target:number):Combination[]{
 if(!Number.isInteger(count)||count<2||count>10||!Number.isFinite(target)||target<2||target>10)return [];
 const groups=new Map<string,ForecastRow[]>();
 for(const r of rows){if(!r.price||r.price.odds<=1||!Number.isFinite(r.price.odds)||!Number.isFinite(r.probability)||r.probability<=0||r.probability>1)continue;const g=groups.get(r.match.id)||[];g.push(r);groups.set(r.match.id,g)}
 const upper=target*1.05,lower=target*.95;
 let beams:Combination[][]=Array.from({length:count+1},()=>[]);beams[0]=[{rows:[],odds:1,probability:1}];
 for(const group of groups.values()){
  for(let n=count;n>=1;n--){
   const buckets=new Map<number,Combination[]>();
   const add=(c:Combination)=>{if(c.odds>upper)return;const key=Math.round(Math.log(c.odds)*200);const b=buckets.get(key)||[];b.push(c);b.sort((a,b)=>b.probability-a.probability);buckets.set(key,b.slice(0,6))};
   beams[n].forEach(add);
   for(const prior of beams[n-1])for(const row of group)add({rows:[...prior.rows,row],odds:prior.odds*row.price!.odds,probability:prior.probability*row.probability});
   beams[n]=[...buckets.values()].flat().sort((a,b)=>b.probability-a.probability).slice(0,3000);
  }
 }
 const ranked=beams[count].filter(c=>c.odds>=lower).sort((a,b)=>b.probability-a.probability||Math.abs(a.odds-target)-Math.abs(b.odds-target));
 const results:Combination[]=[];
 for(const c of ranked){if(results.some(other=>c.rows.filter(r=>other.rows.some(o=>o.match.id===r.match.id)).length>=count-1))continue;results.push(c);if(results.length===3)break}
 return results;
}
