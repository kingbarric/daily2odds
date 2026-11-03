import {Miniflare} from 'miniflare';
import path from 'node:path';
import {createServer} from 'node:http';
import {readFileSync,existsSync,readdirSync,statSync} from 'node:fs';
import {timingSafeEqual} from 'node:crypto';
import {MatchFeed} from './match-feed.mjs';
const files=readdirSync('dist/server',{recursive:true}).filter(f=>/\.m?js$/.test(f));
const mf=new Miniflare({modules:[{type:'ESModule',path:path.resolve('dist/server/index.js'),contents:readFileSync('dist/server/index.js','utf8')},...files.filter(f=>f!=='index.js').map(f=>({type:'ESModule',path:path.resolve('dist/server',f),contents:readFileSync('dist/server/'+f,'utf8')}))],modulesRoot:path.resolve('dist/server'),compatibilityDate:'2026-05-15',compatibilityFlags:['nodejs_compat'],d1Databases:{DB:'daily2odds'},d1Persist:process.env.DAILY_DATA_DIR||'/data/db',assets:{directory:'dist/client',routerConfig:{has_user_worker:true,invoke_user_worker_ahead_of_assets:true},binding:'ASSETS'}});
const db=await mf.getD1Database('DB');
await db.exec('CREATE TABLE IF NOT EXISTS forecast_cache (key TEXT PRIMARY KEY,payload TEXT NOT NULL,updated_at TEXT NOT NULL); CREATE TABLE IF NOT EXISTS subscribers (email TEXT PRIMARY KEY,created_at TEXT NOT NULL,consent TEXT NOT NULL,removal_token TEXT NOT NULL UNIQUE);');
for(const key of ['forecasts','prices','history']){const file=`seed/${key}.json`;if(existsSync(file))await db.prepare('INSERT OR IGNORE INTO forecast_cache(key,payload,updated_at) VALUES (?,?,?)').bind(key,readFileSync(file,'utf8'),new Date().toISOString()).run()}
const feed=new MatchFeed({config:JSON.parse(readFileSync('seed/scoreboard-config.json','utf8')),storage:path.join(process.env.DAILY_DATA_ROOT||'/data','public-feed.json'),archivedPrices:JSON.parse(readFileSync('seed/archived-prices.json','utf8'))});
for(const key of ['history','forecasts','prices']){
 const row=await db.prepare('SELECT payload FROM forecast_cache WHERE key=?').bind(key).first();
 let payload=row?JSON.parse(row.payload):JSON.parse(readFileSync(`seed/${key}.json`,'utf8'));
 if(payload.chunked){const parts=[];for(let n=0;n<payload.count;n++){const part=await db.prepare('SELECT payload FROM forecast_cache WHERE key=?').bind(`${key}:${payload.version}:${n}`).first();if(!part)throw Error('Incomplete cache');parts.push(part.payload)}payload=JSON.parse(parts.join(''))}
 feed.update(key,payload);
}
const poll=()=>feed.poll().catch(error=>console.error('Results poll failed:',error.message));
poll();const pollTimer=setInterval(poll,120000);pollTimer.unref();
const token=process.env.CACHE_IMPORT_TOKEN;
const probeWorker=()=>Promise.race([db.prepare('SELECT 1 AS ok').first(),new Promise((_,reject)=>{const timer=setTimeout(()=>reject(Error('Worker health timeout')),3500);timer.unref()})]);
const server=createServer(async(req,res)=>{try{
 const url=new URL(req.url,'https://daily2odds.playhuud.com');
 if(url.pathname==='/healthz'){try{await probeWorker();res.writeHead(200,{'content-type':'text/plain'});return res.end('ok')}catch{res.writeHead(503);return res.end('Worker unavailable')}}
 if(['GET','HEAD'].includes(req.method)&&['/api/forecasts','/api/history'].includes(url.pathname)){
  res.writeHead(200,{'content-type':'application/json; charset=utf-8','cache-control':'public, max-age=15'});return res.end(req.method==='HEAD'?undefined:url.pathname==='/api/forecasts'?feed.forecastsBody:feed.historyBody);
 }
 if(url.pathname.startsWith('/__cache')){
  const supplied=Buffer.from(req.headers.authorization||'');const expected=Buffer.from(`Bearer ${token}`);
  if(!token||supplied.length!==expected.length||!timingSafeEqual(supplied,expected)||req.method!=='POST'){res.writeHead(404);return res.end()}
  const key=url.searchParams.get('key');if(!['forecasts','prices','history'].includes(key)){res.writeHead(400);return res.end('Invalid key')}
  let raw='';for await(const chunk of req){raw+=chunk;if(raw.length>12000000){res.writeHead(413);return res.end()}}
  const payload=JSON.parse(raw);if(key==='prices'?!payload.matches||Array.isArray(payload.matches):!Array.isArray(payload.matches)){res.writeHead(400);return res.end('Invalid payload')}
  await db.prepare('INSERT INTO forecast_cache(key,payload,updated_at) VALUES (?,?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload,updated_at=excluded.updated_at').bind(key,raw,new Date().toISOString()).run();feed.update(key,payload);res.writeHead(200);return res.end('Saved '+key);
 }
 const assetRoot=path.resolve('dist/client');const assetPath=path.resolve(assetRoot,'.'+decodeURIComponent(url.pathname));
 if(['GET','HEAD'].includes(req.method)&&assetPath.startsWith(assetRoot+path.sep)&&!url.pathname.split('/').some(p=>p.startsWith('.'))&&existsSync(assetPath)&&statSync(assetPath).isFile()){
  const mime={'.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.webp':'image/webp','.woff2':'font/woff2','.ico':'image/x-icon','.json':'application/json','.txt':'text/plain'}[path.extname(assetPath)]||'application/octet-stream';
  res.writeHead(200,{'content-type':mime,'cache-control':url.pathname.startsWith('/_next/static/')?'public, max-age=31536000, immutable':'public, max-age=3600'});return res.end(req.method==='HEAD'?undefined:readFileSync(assetPath));
 }
 const headers=new Headers();for(const [k,v] of Object.entries(req.headers)){if(v)headers.set(k,Array.isArray(v)?v.join(','):v)}
 const chunks=[];let size=0;if(!['GET','HEAD'].includes(req.method))for await(const chunk of req){size+=chunk.length;if(size>65536){res.writeHead(413);return res.end()}chunks.push(chunk)}const body=chunks.length?Buffer.concat(chunks):undefined;
 const response=await mf.dispatchFetch(url.href,{method:req.method,headers,body,redirect:'manual'});res.writeHead(response.status,Object.fromEntries(response.headers));if(response.body)for await(const chunk of response.body)res.write(Buffer.from(chunk));res.end();
}catch(e){console.error('Request failed:',e.message);res.writeHead(500);res.end('Unable to serve request')}});
server.listen(Number(process.env.PORT||3000),'0.0.0.0',()=>console.log('Daily2Odds ready on :'+(process.env.PORT||3000)));
let stopping=false,failures=0;
function stop(code){if(stopping)return;stopping=true;clearInterval(pollTimer);clearInterval(watchdog);setTimeout(()=>process.exit(code),10000).unref();server.close(async()=>{await mf.dispose();process.exit(code)})}
// A dead child does not activate Docker's restart policy unless its parent exits.
const watchdog=setInterval(async()=>{
 try{await probeWorker();failures=0}catch{if(++failures>=2){console.error('Worker unavailable; restarting service');stop(1)}}
 try{const used=Number(readFileSync('/sys/fs/cgroup/memory.current','utf8'));if(used>650*1048576){console.error('Memory safety threshold reached; recycling service');stop(1)}}catch{/* cgroup metrics are Linux-only. */}
},30000);watchdog.unref();
for(const signal of ['SIGTERM','SIGINT'])process.on(signal,()=>stop(0));
