import {env} from 'cloudflare:workers';
export function GET(){const settings=env as unknown as Record<string,string|undefined>;const safe=(s:string|undefined)=>{try{const u=new URL(s||'');return u.protocol==='https:'?u.href:''}catch{return ''}};return Response.json({paypal:safe(settings.PAYPAL_SUPPORT_URL||'https://paypal.me/dailyodds')})}
