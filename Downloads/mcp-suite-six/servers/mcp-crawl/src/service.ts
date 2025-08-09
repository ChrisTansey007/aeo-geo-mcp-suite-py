import * as cheerio from 'cheerio'; import type { JsonRpcReq, JsonRpcRes } from '@mcp/common/transport.js'; import crypto from 'node:crypto';
const sid=(xp:string, payload:string, order:number)=>{const h=crypto.createHash('sha1').update(xp+'|'+payload).digest('hex').slice(0,4); return `b-${String(order).padStart(4,'0')}-${h}`;}
function xpath($:cheerio.CheerioAPI, el:cheerio.Element){const parts:string[]=[]; let n:any=el; while(n&&n.type!=='root'){const name=n.name||n.tagName||n.type; const sib=n.parent? n.parent.children?.filter((c:any)=>c.type==='tag'&&c.name===name):[]; const idx=sib? sib.indexOf(n)+1:1; parts.unshift(`/${name}${idx?`[${idx}]`:''}`); n=n.parent;} return parts.join('');}
export async function handle(req: JsonRpcReq): Promise<JsonRpcRes>{
  const id=req.id??null;
  try{
    if(req.method==='crawl.fetch'){ const {url}=req.params||{}; const res=await fetch(url); const text=await res.text();
      return {jsonrpc:'2.0',id,result:{content:text,metadata:{final_url:res.url,mime:res.headers.get('content-type')||'text/html',fetched_at:new Date().toISOString()}}}; }
    if(req.method==='crawl.canon'){ const {html}=req.params||{}; const $=cheerio.load(html||''); const blocks:any[]=[]; let order=1;
      $('h1,h2,h3,h4,h5,h6,p,ul,ol,table').each((_,el)=>{const name=el.tagName.toLowerCase(); const xp=xpath($,el);
        if(name==='table'){const cells:string[][]=[]; $(el).find('tr').each((_,tr)=>{ const row:string[]=[]; $(tr).find('th,td').each((_,td)=>row.push($(td).text().trim())); if(row.length) cells.push(row);}); blocks.push({id:sid(xp,JSON.stringify(cells),order++),type:'table',cells,xpath:xp});}
        else if(name==='ul'||name==='ol'){const txt=$(el).find('li').toArray().map(li=>$(li).text().trim()).join('\n'); blocks.push({id:sid(xp,txt,order++),type:'list',text:txt,xpath:xp});}
        else if(name==='p'){const txt=$(el).text().trim(); if(txt) blocks.push({id:sid(xp,txt,order++),type:'para',text:txt,xpath:xp});}
        else {const txt=$(el).text().trim(); blocks.push({id:sid(xp,txt,order++),type:'heading',text:txt,xpath:xp});}});
      return {jsonrpc:'2.0',id,result:{blocks}}; }
    return {jsonrpc:'2.0',id,error:{code:'method_not_found',message:`Unknown method ${req.method}`}};
  }catch(e:any){ return {jsonrpc:'2.0',id,error:{code:'internal',message:String(e?.message||e)}}; }
}
