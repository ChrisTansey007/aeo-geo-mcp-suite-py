import http from 'node:http'; import { registry } from './metrics.js';
export type JsonRpcReq={jsonrpc:'2.0',id:number|string|null,method:string,params?:any};
export type JsonRpcRes={jsonrpc:'2.0',id:number|string|null,result?:any,error?:any};
function token(h:any){const b=(h['authorization']||'').toString(); const m=b.match(/^Bearer\s+(.+)$/i); return m?m[1]:undefined;}
export function startHttp(service:string, handle:(r:JsonRpcReq)=>Promise<JsonRpcRes>, port=3000){
  const s=http.createServer(async (req,res)=>{
    if(req.url==='/metrics'){res.writeHead(200,{'Content-Type':'text/plain'});res.end(await registry.metrics());return;}
    if(req.url==='/healthz'){res.writeHead(200);res.end('ok');return;}
    if(req.method!=='POST'||req.url!=='/rpc'){res.statusCode=404;res.end();return;}
    const key=token(req.headers); if(!check(key)){res.statusCode=401;res.end(JSON.stringify({error:{code:'unauthorized'}}));return;}
    let b=''; for await (const c of req) b+=c;
    try{const j=JSON.parse(b); const out=await handle(j); res.setHeader('Content-Type','application/json'); res.end(JSON.stringify(out));}
    catch(e:any){res.statusCode=400;res.end(JSON.stringify({jsonrpc:'2.0',id:null,error:{code:'bad_request',message:String(e?.message||e)}}));}
  }); s.listen(port,()=>console.log(`[${service}] :${port}`)); return s;
  function check(k?:string){const env=process.env.MCP_API_KEY; if(!env) return true; return k===env;}
}
export function startStdio(handle:(r:JsonRpcReq)=>Promise<JsonRpcRes>){
  process.stdin.setEncoding('utf8'); let buf='';
  process.stdin.on('data', async (chunk)=>{ buf+=chunk; let i; while((i=buf.indexOf('\n'))>=0){ const line=buf.slice(0,i); buf=buf.slice(i+1);
      try{const j=JSON.parse(line); const out=await handle(j); process.stdout.write(JSON.stringify(out)+'\n');}
      catch(e:any){process.stdout.write(JSON.stringify({jsonrpc:'2.0',id:null,error:{code:'bad_request',message:String(e?.message||e)}})+'\n');} } });
}
