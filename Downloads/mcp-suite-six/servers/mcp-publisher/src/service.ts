import type { JsonRpcReq, JsonRpcRes } from '@mcp/common/transport.js'; import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'; import { dirname, join } from 'node:path';
function repo(){return process.env.PUBLISHER_REPO || './content-repo';}
function urlFor(p:string){return (process.env.PUBLISHER_BASE_URL||'https://example.com').replace(/\/$/,'')+p.replace(/\.md$/,'/').replace(/\.html$/,'/');}
function readSafe(p:string){try{return readFileSync(join(repo(),p.replace(/^\/+/,'')),'utf8');}catch{return '';}}
function writeFileRel(p:string, data:string){const abs=join(repo(),p.replace(/^\/+/,'')); mkdirSync(dirname(abs),{recursive:true}); writeFileSync(abs,data,'utf8');}
function diff(oldS:string,newS:string,p:string){ if(oldS===newS) return ''; const o=oldS.split(/\r?\n/), n=newS.split(/\r?\n/); let d=`--- a${p}\n+++ b${p}\n@@\n`; for(const l of o) d+=`-${l}\n`; for(const l of n) d+=`+${l}\n`; return d; }
export async function handle(req: JsonRpcReq): Promise<JsonRpcRes>{ const id=req.id??null; try{
  if(req.method==='publish.upsert'){ const {path,markdown,html,jsonld,dry_run=false}=req.params||{}; const payload=markdown??html??''; const before=readSafe(path); const d=diff(before,payload,path); if(!dry_run){ writeFileRel(path,payload); if(jsonld) writeFileRel(path.replace(/\.(md|html)$/i,'')+'.jsonld', JSON.stringify(jsonld,null,2)); } return {jsonrpc:'2.0',id,result:{status:'ok',diff:d,url:urlFor(path),version:'v0'}}; }
  if(req.method==='publish.rollback'){ const {path}=req.params||{}; return {jsonrpc:'2.0',id,result:{status:'ok',url:urlFor(path)}}; }
  if(req.method==='publish.ping'){ return {jsonrpc:'2.0',id,result:{ok:true}}; }
  return {jsonrpc:'2.0',id,error:{code:'method_not_found',message:`Unknown method ${req.method}`}};
 }catch(e:any){ return {jsonrpc:'2.0',id,error:{code:'internal',message:String(e?.message||e)}}; } }
