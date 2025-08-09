import Handlebars from 'handlebars'; import type { JsonRpcReq, JsonRpcRes } from '@mcp/common/transport.js';
const TPLS: Record<string,string> = {
  LocalBusiness: '{ "@context":"https://schema.org", "@type":["LocalBusiness"{{#if subType}},"{{subType}}"{{/if}}], "name":"{{name}}" {{#if url}},"url":"{{url}}"{{/if}} }',
  Product: '{ "@context":"https://schema.org", "@type":"Product", "name":"{{name}}" }',
  Article: '{ "@context":"https://schema.org", "@type":"Article", "headline":"{{headline}}" {{#if datePublished}},"datePublished":"{{datePublished}}"{{/if}} }'
};
export async function handle(req: JsonRpcReq): Promise<JsonRpcRes>{ const id=req.id??null; try{
  if(req.method==='schema.generate'){ const {type,data}=req.params||{}; const tpl=Handlebars.compile(TPLS[type]||'{}',{noEscape:true}); const jsonld=JSON.parse(tpl(data||{})); return {jsonrpc:'2.0',id,result:{jsonld,warnings:[]}}; }
  if(req.method==='schema.validate'){ const {jsonld}=req.params||{}; const valid=!!(jsonld&&jsonld['@context']&&jsonld['@type']); return {jsonrpc:'2.0',id,result:{valid,errors: valid?[]:[{path:'/','message':'@context and @type required','rule':'basic'}],suggestions:[]}}; }
  return {jsonrpc:'2.0',id,error:{code:'method_not_found',message:`Unknown method ${req.method}`}};
 }catch(e:any){ return {jsonrpc:'2.0',id,error:{code:'internal',message:String(e?.message||e)}}; } }
