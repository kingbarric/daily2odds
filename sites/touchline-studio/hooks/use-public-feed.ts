'use client';
import {useEffect,useState} from 'react';
type State<T>={data:T|null;error:string;refreshing:boolean;receivedAt:number};
type Entry={state:State<unknown>;listeners:Set<()=>void>;timer?:ReturnType<typeof setInterval>};
const feeds=new Map<string,Entry>();
function getEntry(url:string){let entry=feeds.get(url);if(!entry){entry={state:{data:null,error:'',refreshing:false,receivedAt:0},listeners:new Set()};feeds.set(url,entry)}return entry}
async function refresh(url:string){
 const entry=getEntry(url);if(entry.state.refreshing)return;
 const notify=()=>entry.listeners.forEach(listener=>listener());entry.state={...entry.state,refreshing:true};notify();
 try{const response=await fetch(url,{cache:'no-store',signal:AbortSignal.timeout(15000)});if(!response.ok)throw Error();const data=await response.json();entry.state={data,error:'',refreshing:false,receivedAt:Date.now()}}
 catch{entry.state={...entry.state,error:'Updates are delayed. Showing the last available data; we’ll retry automatically.',refreshing:false}}notify();
}
/** Share one poll per endpoint across sections, retaining the last successful data on failures. */
export function usePublicFeed<T>(url:string){
 const [state,setState]=useState<State<T>>({data:null,error:'',refreshing:false,receivedAt:0});
 useEffect(()=>{
  const entry=getEntry(url),listener=()=>setState(entry.state as State<T>);entry.listeners.add(listener);listener();
  const visibleRefresh=()=>{if(document.visibilityState!=='hidden'&&Date.now()-entry.state.receivedAt>10000)void refresh(url)};
  if(!entry.timer)entry.timer=setInterval(visibleRefresh,60000);
  if(!entry.state.data||Date.now()-entry.state.receivedAt>60000)void refresh(url);
  window.addEventListener('focus',visibleRefresh);document.addEventListener('visibilitychange',visibleRefresh);
  return()=>{entry.listeners.delete(listener);window.removeEventListener('focus',visibleRefresh);document.removeEventListener('visibilitychange',visibleRefresh);if(!entry.listeners.size){clearInterval(entry.timer);entry.timer=undefined}};
 },[url]);
 return {...state,loading:!state.data&&!state.error,refresh:()=>refresh(url)};
}
