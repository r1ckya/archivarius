import React, {useState} from 'react';
import {useLocation} from "react-router-dom";
import {Button, Col, Form, FormControl, Nav, Navbar, NavDropdown, Row, Table} from "react-bootstrap";
import ResultTable from "../Modules/ResultsTable";
import APIRequest from "../../rest";
import ClassesSelect from "../Modules/ClassesSelect";

function useQuery() {
  return new URLSearchParams(useLocation().search);
}


let Search = function (props) {
    const [state, setState] = useState({}); // todo: get state from query
    const [result, setResult] = useState(null);
    const [classes, setClasses] = useState(null);
    let handleChange = function(event) {
      const target = event.target;
      const name = target.name;
      console.log(target);
      if (name !== 'file') {
        setState({
          [name]: target.value
        });
      } else {
        setState({
          [name]: target.files[0]
        });
      }
    }
    let onSubmit = function(e) {
      e.preventDefault();
      APIRequest('search', state).then(function (result) {
        console.log(result);
        if (result.status === 'ok') {
          setResult(result);
        }
      });

    }

    let query = useQuery();
    return (
      <div>
        <h1 className={'mt-2'}> Поиск </h1>
      <hr/>
      <Form method={"GET"} onSubmit={onSubmit}>
        <Form.Group as={Row} controlId="exampleForm.ControlInput1">
          <Form.Label column sm="2">Текст</Form.Label>
          <Col sm="10">
            <Form.Control onChange={handleChange} type="text" name={'text'} placeholder="Документ содержит..." value={query.get('text')} />
          </Col>
        </Form.Group>
        <ClassesSelect classes={props.classes} handleChange={handleChange} type={query.get('doc_cls')}/>
        {/*<Row>
        <Form.Group  as={Col} column sm="5" controlId="exampleForm.DateFrom">

          <Form.Label>Дата подписания от</Form.Label>
            <Form.Control onChange={handleChange} name={'dateBegin'} type="date"></Form.Control>

        </Form.Group>
          <Form.Group as={Col} column sm="5" controlId="exampleForm.DateTo">
            <Form.Label>До</Form.Label>
            <Form.Control  onChange={handleChange} name={'dateEnd'} type="date"></Form.Control>
          </Form.Group>
        </Row>*/}

        <Button type={"submit"} variant="outline-success">Искать</Button>
      </Form>
        <hr/>
        { (result != null) ? <ResultTable result={result}/> : null }
      </div>
    )

}

export default Search;