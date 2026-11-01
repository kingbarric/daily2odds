import dataset from '../../../data/forecasts.json';
import fallback from '../../../data/prices.json';
import archived from '../../../data/archived-prices.json';
import {getDb} from '../../../db';
import {forecastCache} from '../../../db/schema';
import {inArray} from 'drizzle-orm';
import type {Match} from '../../../lib/forecast-view';
type PriceCache={capturedAt:string;coverage:string;matches:Record<string,Record<string,unknown>>};
type ForecastData={snapshot:string;matches:Match[];windowStart?:string;windowEnd?:string;generatedAt?:string;teamNews?:{status:string;reason:string;applied:boolean};leagues?:Record<string,unknown>;status?:string};
export async function GET(){
 let prices:PriceCache=fallback;let forecasts=dataset as unknown as ForecastData;
 try{
  const db=getDb();const rows=await db.select().from(forecastCache).where(inArray(forecastCache.key,['prices','forecasts']));
  for(const row of rows){
    let value=JSON.parse(row.payload);
    if(value.chunked&&typeof value.version==='string'&&Number.isInteger(value.count)&&value.count>0&&value.count<100){
      const keys=Array.from({length:value.count},(_,i)=>`${row.key}:${value.version}:${i}`);
      const parts=await db.select().from(forecastCache).where(inArray(forecastCache.key,keys));
      const byKey=new Map(parts.map(p=>[p.key,p.payload]));
      if(!keys.every(k=>byKey.has(k)))throw Error('Incomplete saved snapshot');
      value=JSON.parse(keys.map(k=>byKey.get(k)).join(''));
    }
    if(row.key==='prices')prices=value;
    else if(Array.isArray(value.matches))forecasts=value;
  }
 }catch{/* Keep the last checked-in snapshot available if the cache is unavailable. */}
 // Saved outputs only: no provider calls, training or per-visitor model fitting.
 const matches=forecasts.matches.map(m=>({...m,prices:{...(archived.matches as Record<string,Record<string,unknown>>)[m.id],...m.prices,...prices.matches[m.id]}}));
 return Response.json({...forecasts,matches,priceCoverage:prices.coverage,pricesUpdatedAt:prices.capturedAt},{headers:{'Cache-Control':'public, max-age=30'}})
}
