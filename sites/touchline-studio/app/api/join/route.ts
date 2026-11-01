import {getDb} from '../../../db';
import {subscribers} from '../../../db/schema';
export async function POST(request:Request){
 const headers={'Cache-Control':'no-store'};
 if(request.headers.get('origin')!==new URL(request.url).origin)return Response.json({error:'Please submit the form from this website.'},{status:403,headers});
 if(!request.headers.get('content-type')?.includes('application/json'))return Response.json({error:'JSON required'},{status:415,headers});
 if(Number(request.headers.get('content-length')||0)>2048)return Response.json({error:'Request too large'},{status:413,headers});
 try{const raw=await request.text();if(raw.length>2048)return Response.json({error:'Request too large'},{status:413,headers});const body=JSON.parse(raw);const email=typeof body.email==='string'?body.email.trim().toLowerCase():'';
 if(!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)||email.length>254||body.consent!==true)return Response.json({error:'Enter a valid email and agree to the optional updates.'},{status:400,headers});
 const token=crypto.randomUUID();const result=await getDb().insert(subscribers).values({email,createdAt:new Date().toISOString(),consent:'Daily2Odds project updates only; privacy-v1',removalToken:token}).onConflictDoNothing().returning({email:subscribers.email});
 return Response.json({message:'Your interest has been registered. The explorer is always free. No confirmation email is sent.',...(result.length?{removalUrl:`/api/unsubscribe?token=${encodeURIComponent(token)}`}:{})},{headers});
 }catch{return Response.json({error:'Signup could not be saved. Please try again later.'},{status:503,headers})}
}
