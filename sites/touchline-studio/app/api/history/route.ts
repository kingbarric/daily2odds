import fallback from '../../../data/history.json';
import {getDb} from '../../../db';
import {forecastCache} from '../../../db/schema';
import {inArray} from 'drizzle-orm';
export async function GET(){
 let history=fallback;
 try{const db=getDb();const [row]=await db.select().from(forecastCache).where(inArray(forecastCache.key,['history']));if(row){let value=JSON.parse(row.payload);if(value.chunked&&typeof value.version==='string'&&Number.isInteger(value.count)&&value.count>0&&value.count<100){const keys=Array.from({length:value.count},(_,i)=>`history:${value.version}:${i}`);const parts=await db.select().from(forecastCache).where(inArray(forecastCache.key,keys));const map=new Map(parts.map(p=>[p.key,p.payload]));if(!keys.every(k=>map.has(k)))throw Error('Incomplete history');value=JSON.parse(keys.map(k=>map.get(k)).join(''))}if(Array.isArray(value.matches))history=value}}catch{/* Retain archived local data. */}
 return Response.json(history,{headers:{'Cache-Control':'public, max-age=60'}})
}
